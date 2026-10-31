#!/usr/bin/env python3
"""Reassign Qwen's last 1000 base-vocabulary IDs to Rex-Omni coordinates."""
import argparse
import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source, output = Path(args.model), Path(args.output)
    obj = json.loads((source / 'tokenizer.json').read_text(encoding='utf-8'))
    config = json.loads((source / 'tokenizer_config.json').read_text(encoding='utf-8'))
    vocab = obj['model']['vocab']
    reverse = {idx: token for token, idx in vocab.items()}
    first, last = 150643, 151642
    ids = range(first, last + 1)
    if len(reverse) != len(vocab) or not all(i in reverse for i in ids):
        raise ValueError('Qwen3 base vocabulary does not match the expected Rex-Omni ID range')
    if any(first <= item['id'] <= last for item in obj['added_tokens']):
        raise ValueError('Rex-Omni coordinate range overlaps an existing added token')
    removed = {reverse[i] for i in ids}
    if any(f'<{i}>' in vocab for i in range(1000)):
        raise ValueError('coordinate tokens already exist in base vocabulary')
    old_merges = obj['model']['merges']
    filtered = []
    for merge in old_merges:
        pieces = merge.split(' ') if isinstance(merge, str) else merge
        if len(pieces) != 2:
            raise ValueError(f'unrecognized BPE merge: {merge!r}')
        left, right = pieces
        if left not in removed and right not in removed and left + right not in removed:
            filtered.append(merge)
    for i in ids:
        del vocab[reverse[i]]
        coord = f'<{i - first}>'
        vocab[coord] = i
        token = {'id': i, 'content': coord, 'single_word': False, 'lstrip': False,
                 'rstrip': False, 'normalized': False, 'special': True}
        obj['added_tokens'].append(token)
        config.setdefault('added_tokens_decoder', {})[str(i)] = {k: v for k, v in token.items() if k != 'id'}
    obj['model']['merges'] = filtered
    config.setdefault('additional_special_tokens', []).extend(f'<{i}>' for i in range(1000))
    tokenizer_text = json.dumps(obj, ensure_ascii=False)
    tok = Tokenizer.from_str(tokenizer_text)
    for i in range(1000):
        token = f'<{i}>'
        if tok.encode(token, add_special_tokens=False).ids != [first + i]:
            raise ValueError(f'coordinate token mapping failed: {token}')
    for token in ('<|im_end|>', '<|image_pad|>', '<|object_ref_start|>', '<|box_start|>'):
        original = next(x['id'] for x in obj['added_tokens'] if x['content'] == token)
        if tok.token_to_id(token) != original:
            raise ValueError(f'structural token changed: {token}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tokenizer.json').write_text(tokenizer_text, encoding='utf-8')
    (output / 'tokenizer_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name in ('preprocessor_config.json', 'video_preprocessor_config.json', 'chat_template.json', 'config.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    manifest = {'source': str(source), 'coordinate_ids': [first, last], 'removed_bpe_tokens': len(removed),
                'removed_bpe_merges': len(old_merges) - len(filtered), 'vocab_size': len(vocab),
                'rex_reference': 'IDEA-Research/Rex-Omni@c1d245b31bdc8159293a55573913afcea396a15b'}
    (output / 'rex_id_mapping.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
