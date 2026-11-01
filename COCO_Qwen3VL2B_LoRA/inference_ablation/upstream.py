"""Load audited pure fragments from fixed author source revisions, not their trainers."""
import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import find_objects, label


class Upstream:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.zoom = self._functions('zoom.py', ['compute_crop_box', 'zoom_consistency', 'remap'], {'math': math})
        self._build_vote()
        self._build_mcc()

    def source(self, name):
        return ast.parse((self.folder / name).read_text(encoding='utf-8'))

    def _functions(self, name, names, scope):
        nodes = [n for n in self.source(name).body if isinstance(n, ast.FunctionDef) and n.name in names]
        if len(nodes) != len(names):
            raise ValueError(f'Official source functions changed: {name}')
        scope = dict(scope, CROP_RATIO=0.5)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
        return scope

    def _build_vote(self):
        # Reuse the author's max-vote connected-component extraction exactly.
        # Drop only the final comparison against gt_bbox (an evaluation metric,
        # never an input to inference). No author model-loading code is executed.
        matches = [n for n in ast.walk(self.source('gui_rc_evaluation.py')) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'max_votes > 0']
        if len(matches) != 1:
            raise ValueError('Official voting block changed')
        block = matches[0]
        class RemoveMetric(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'consensus_region_center is not None':
                    return None
                return self.generic_visit(node)

        block = RemoveMetric().visit(block)
        if any(isinstance(n, ast.Name) and n.id == 'gt_bbox' for n in ast.walk(block)):
            raise ValueError('Ground truth leaked into voting implementation')
        self.vote_code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                                 'official GUI-RC consensus fragment', 'exec')

    def vote(self, boxes, width, height):
        grid = torch.zeros((height, width), dtype=torch.int32)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            grid[y1:y2, x1:x2] += 1
        scope = {'grid': grid, 'max_votes': grid.max().item(), 'consensus_region': None,
                 'label': label, 'find_objects': find_objects}
        exec(self.vote_code, scope)
        return scope['consensus_region'], scope['max_votes']

    def _build_mcc(self):
        tree = self.source('mcc_qwen3_2B_grefcoco.py')
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        count = [n.value.elt for n in assignments if any(isinstance(t, ast.Name) and t.id == 'count_qs' for t in n.targets)]
        arb = [n.value for n in assignments if any(isinstance(t, ast.Name) and t.id == 'arb_q' for t in n.targets)]
        if len(count) != 1 or len(arb) != 1:
            raise ValueError('Official MCC prompt construction changed')
        self.count_code = compile(ast.Expression(count[0]), 'official MCC counting prompt', 'eval')
        self.arb_code = compile(ast.Expression(arb[0]), 'official MCC recheck prompt', 'eval')

    def count_prompt(self, target):
        return eval(self.count_code, {'item': {'refer_text': target}})

    def recheck_prompt(self, target, expected, predictions):
        return eval(self.arb_code, {'exp_count': expected, 'initial_preds': predictions,
                                    'valid_items': [{'refer_text': target}], 'idx': 0})
