"""Add object and termination credit to upstream Swift's token advantages."""
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from credit import ObjectAnnotator, assign_group_credit


def response_frame_positions(raw_ids, frame_ids, completion_mask):
    active_positions = completion_mask.nonzero(as_tuple=False).flatten().tolist()
    active_ids = frame_ids[completion_mask].tolist()
    # Match IDs, never retokenize the decoded response. Extra template tokens
    # may be masked-in, but all credited positions must belong to the actual
    # sampled response and be active in the real GRPO loss mask.
    starts = [i for i in range(len(active_ids) - len(raw_ids) + 1)
              if active_ids[i:i + len(raw_ids)] == raw_ids]
    if not raw_ids or len(starts) != 1:
        raise ValueError(f'Actual response does not align uniquely with training tokens: '
                         f'raw={raw_ids[:12]} ({len(raw_ids)}), active={active_ids[:12]} ({len(active_ids)})')
    start = starts[0]
    return active_positions[start:start + len(raw_ids)]


def install(trainer_class, root, settings_path):
    root = Path(root)
    settings = json.loads(Path(settings_path).read_text())
    sys.path.insert(0, str(root / 'rex_id_grpo'))
    from reward_plugin import core, parse

    metadata = json.loads((root / 'rex_id_object_credit/ignore_metadata.json').read_text())
    ignore_by_image = metadata['images']
    original = trainer_class._postprocess_batch

    def postprocess(self, samples, batch_encoded_inputs):
        original(self, samples, batch_encoded_inputs)
        if self.accelerator.num_processes != 1 or self.template.padding_free or self.dynamic_num_samples:
            raise ValueError('Object credit v1 requires single-GPU padded single-turn grouped GRPO')
        if not hasattr(self, '_object_annotator'):
            self._object_annotator = ObjectAnnotator(self.template.tokenizer, parse, settings)
            output = Path(self.args.output_dir)
            (output / 'object_credit_config.json').write_text(json.dumps(settings, indent=2) + '\n')
        annotator = self._object_annotator
        training = self.model.training
        mode = 'train' if training else 'eval'
        groups = defaultdict(list)
        annotations = []
        truths = []
        for i, sample in enumerate(samples):
            image_id = str(sample.extra['image_id'])
            if image_id not in ignore_by_image:
                raise ValueError(f'Image {image_id} is outside the fixed credit metadata')
            truth = core._parse_truth(sample.extra['ground_truth'])
            annotation = annotator.annotate(sample.response_token_ids, sample.finish_reason, truth,
                                            ignore_by_image[image_id])
            annotations.append(annotation)
            truths.append(truth)
            groups[image_id].append(i)
        if training:
            if len(groups) != 32 or len(samples) != 256:
                raise ValueError('Object credit must be computed on the complete 32 x 8 rollout')
            for image_id, indices in groups.items():
                if any(truths[i] != truths[indices[0]] for i in indices):
                    raise ValueError(f'Ground-truth identity differs within image {image_id}')
                assign_group_credit([annotations[i] for i in indices], settings)

        n_gt = sum(len(a.quality) for a in annotations)
        n_pred = sum(len(a.predictions) for a in annotations)
        n_missing = sum(len(a.missing) for a in annotations)
        n_extra = sum(p.gt_index is None and not p.ignored for a in annotations for p in a.predictions)
        metrics = self._metrics[mode]
        metrics['object_credit/quality_per_gt'].append(sum(sum(a.quality) for a in annotations) / max(n_gt, 1))
        metrics['object_credit/missing_gt_fraction'].append(n_missing / max(n_gt, 1))
        metrics['object_credit/extra_prediction_fraction'].append(n_extra / max(n_pred, 1))
        metrics['object_credit/valid_response_fraction'].append(sum(a.valid for a in annotations) / len(annotations))
        if not training:
            return  # Validation records diagnostics; its advantages/loss stay unchanged.

        chunks = self.split_by_mini_batches(samples)
        offset = 0
        records = []
        extra_positive = 0
        stop_count = 0
        for chunk, encoded in zip(chunks, batch_encoded_inputs):
            batch = encoded['grpo_batch']
            frame_ids = encoded['model_inputs']['input_ids'][:, -batch.logits_to_keep:]
            if batch.advantages.shape != batch.completion_mask.shape:
                raise ValueError('Expected the upstream [B,T] advantage tensor')
            delta = torch.zeros_like(batch.advantages)
            for row, sample in enumerate(chunk):
                i = offset + row
                if sample is not samples[i]:
                    raise ValueError('Sample order changed during microbatch splitting')
                annotation = annotations[i]
                if not annotation.valid:
                    continue
                frame_positions = response_frame_positions(annotation.response_ids, frame_ids[row],
                                                           batch.completion_mask[row])
                for pred in annotation.predictions:
                    positions = [frame_positions[index] for index in pred.token_indices]
                    delta[row, positions] += settings['box_weight'] * pred.credit
                    if pred.gt_index is None and not pred.ignored:
                        extra_positive += int((batch.advantages[row, positions[0]] + delta[row, positions[0]]) > 0)
                for index, credit, reason in annotation.stops:
                    position = frame_positions[index]
                    token_id = annotation.response_ids[index]
                    if reason.startswith('class_stop:'):
                        if token_id != annotator.marker_ids['<|box_end|>']:
                            raise ValueError('Class stopping penalty is not on an actual box_end token')
                    elif token_id not in annotator.eos_ids:
                        raise ValueError('EOS penalty is not on a sampled EOS token')
                    delta[row, position] += settings['stop_weight'] * credit
                    stop_count += 1
                records.append({'step': self.state.global_step + 1, 'image_id': sample.extra['image_id'],
                                'request_id': sample.request_id, 'base_advantage': float(sample.advantages),
                                'annotation': asdict(annotation)})
            if torch.any(delta[~batch.completion_mask] != 0):
                raise ValueError('Credit leaked into prompt/padding tokens')
            batch.advantages = batch.advantages + delta
            if not torch.isfinite(batch.advantages).all():
                raise ValueError('Non-finite object-credit advantage')
            offset += len(chunk)
        if offset != len(samples):
            raise ValueError('Incomplete credit-to-microbatch mapping')
        metrics['object_credit/extra_final_positive_fraction'].append(extra_positive / max(n_extra, 1))
        metrics['object_credit/penalized_stop_tokens'].append(stop_count)
        with (Path(self.args.output_dir) / 'object_credit_audit.jsonl').open('a') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        if not getattr(self, '_object_credit_verified', False):
            print(f'OBJECT_CREDIT_VERIFIED: 32 images x 8 responses; actual token-ID alignment; '
                  f'{n_pred} boxes; {n_extra} extras; {stop_count} penalized actual stop tokens; '
                  f'box_weight={settings["box_weight"]}, stop_weight={settings["stop_weight"]}', flush=True)
            self._object_credit_verified = True

    trainer_class._postprocess_batch = postprocess
