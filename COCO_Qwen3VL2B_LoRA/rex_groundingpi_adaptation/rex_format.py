"""Strict parser for Qwen3-VL SFT's Rex marker/coordinate answer format."""
import re

_ENTRY = re.compile(r'<\|object_ref_start\|>([^<>]+)<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>', re.S)
_COORD = r'<(\d{1,3})>'
_BOX = re.compile(r'\s*' + r'\s*'.join([_COORD] * 4) + r'\s*')
_END = re.compile(r'\s*<\|im_end\|>\s*$')


def parse_rex_boxes(text):
    """Return (list[(label, normalized 0..1000 box)], valid_format).

    Coordinates use the SFT serializer's integer bins 0..999. Reject incomplete,
    malformed, out-of-range or reversed boxes rather than silently repairing.
    A complete top-level `None` and a category-specific `None` are valid.
    An EOS suffix is optional because rollout implementations may strip it.
    """
    if not isinstance(text, str):
        return [], False
    text = _END.sub('', text).strip()
    if text == 'None':
        return [], True
    if not text:
        return [], False
    items, position = [], 0
    while position < len(text):
        match = _ENTRY.match(text, position)
        if not match:
            return [], False
        label = match.group(1).strip().lower()
        payload = match.group(2).strip()
        if not label or not payload:
            return [], False
        if payload != 'None':
            for group in payload.split(','):
                box = _BOX.fullmatch(group)
                if not box:
                    return [], False
                x1, y1, x2, y2 = map(int, box.groups())
                if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                    return [], False
                items.append((label, [v * 1000. / 999. for v in (x1, y1, x2, y2)]))
        position = match.end()
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        if text[position] != ',':
            return [], False
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return [], False
    return items, True
