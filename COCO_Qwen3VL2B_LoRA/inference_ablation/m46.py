"""Reuse author MCC counting/recheck prompts and nonempty-result fallback."""
import re

from common import FORMAT, TARGET, detections, parse


def run(engine, image, info, baseline, config, upstream, categories):
    settings = config['methods']['m46']
    count = engine.generate(image, upstream.count_prompt(TARGET), f'm46:count:{info["id"]}',
                            max_tokens=settings['count_max_new_tokens'])[0]
    text = count['text'].replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    match = re.search(r'\d+', text)
    expected = int(match.group()) if match else settings['fallback_count_on_parse_error']
    valid_integer = bool(re.fullmatch(r'\d+', text))
    conflict = len(baseline) != expected
    final = baseline
    recheck_count = None
    if conflict:
        prompt = upstream.recheck_prompt(TARGET, expected, baseline) + FORMAT
        result = engine.generate(image, prompt, f'm46:recheck:{info["id"]}')[0]
        refined = parse(result, engine.tokenizer, info['width'], info['height'])
        recheck_count = len(refined)
        if refined:  # Same fallback as the public Qwen3-2B script.
            final = refined
    return {'mcc': detections(final, categories)}, {'counter_text': text, 'expected_count': expected,
            'strict_integer_response': valid_integer, 'initial_count': len(baseline),
            'conflict': conflict, 'recheck_count': recheck_count, 'final_count': len(final)}
