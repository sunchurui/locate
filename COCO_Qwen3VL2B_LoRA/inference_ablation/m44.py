"""User-approved single-model, per-object zoom-consistency adaptation."""
from PIL import Image

from common import FORMAT, detections, iou, parse


def run(engine, image, info, baseline, config, upstream, categories):
    settings = config['methods']['m44']
    always, gated = [], []
    details = []
    width, height = image.size
    for index, initial in enumerate(baseline):
        x1, y1, x2, y2 = initial['box']
        crop_box = upstream.zoom['compute_crop_box']((x1 + x2) / 2, (y1 + y2) / 2, width, height,
                                                       settings['crop_ratio'])
        cropped = image.crop(crop_box).resize(image.size, Image.Resampling.LANCZOS)
        prompt = f'Locate the {initial["label"]} nearest the center of this image.' + FORMAT
        result = engine.generate(cropped, prompt, f'm44:{info["id"]}:{index}')[0]
        candidates = parse(result, engine.tokenizer, width, height)
        cropped.close()
        proposals = []
        for candidate in candidates:
            if candidate['label'] != initial['label']:
                continue
            bx1, by1, bx2, by2 = candidate['box']
            center = ((bx1 + bx2) / 2 / width * 1000, (by1 + by2) / 2 / height * 1000)
            consistency = upstream.zoom['zoom_consistency'](center)
            a = upstream.zoom['remap']((bx1 / width * 1000, by1 / height * 1000), crop_box, width, height)
            b = upstream.zoom['remap']((bx2 / width * 1000, by2 / height * 1000), crop_box, width, height)
            proposal = dict(candidate, box=[a[0] * width, a[1] * height, b[0] * width, b[1] * height])
            proposals.append((iou(initial['box'], proposal['box']), consistency, proposal))
        if proposals:
            overlap, consistency, proposal = max(proposals, key=lambda item: (item[0], -item[1]))
        else:
            overlap, consistency, proposal = 0., None, initial
        eligible = overlap >= settings['candidate_match_iou']
        accepted = eligible and consistency is not None and consistency <= settings['consistency_threshold']
        always.append(proposal if eligible else initial)
        gated.append(proposal if accepted else initial)
        details.append({'box_index': index, 'crop_box': crop_box, 'association_iou': overlap,
                        'consistency': consistency, 'accepted': accepted, 'candidate_count': len(proposals)})
    return {'zoom_always': detections(always, categories), 'zoom_consistency': detections(gated, categories)}, details
