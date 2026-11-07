"""CPU-only reward and format checks; no model imports or external dependencies."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IoU:
    def calculate_iou(self, a, b):
        inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter
        return inter/union if union else 0.


for name in ('verl', 'verl.configs'):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
stub = types.ModuleType('verl.configs.reward_func')
stub.BoxIoURewardFunction = IoU
sys.modules[stub.__name__] = stub
base = load('verl.configs.groundingpi_reward', HERE / 'groundingpi_reward_reference.py')
parser = load('verl.configs.rex_format', HERE / 'rex_format.py')
adapted = load('rex_groundingpi_reward', HERE / 'rex_groundingpi_reward.py')


class RewardChecks(unittest.TestCase):
    def test_matches_original_json_reward(self):
        text = ('<|object_ref_start|>cat<|object_ref_end|><|box_start|>'
                '<100><100><300><300>,<110><110><290><290><|box_end|><|im_end|>')
        pred, valid = parser.parse_rex_boxes(text)
        self.assertTrue(valid)
        truth = [{'label': 'cat', 'bbox_2d': [100, 100, 300, 300]}]
        native = adapted.compute_score([text], [truth])[0]
        baseline = base.compute_score([json.dumps([{'label': l, 'bbox_2d': b} for l, b in pred])], [truth])[0]
        for key in ('overall', 'groundingpi_set', 'groundingpi_strict'):
            self.assertAlmostEqual(native[key], baseline[key])

    def test_none_and_multiple_categories(self):
        empty = adapted.compute_score(['None<|im_end|>'], [[]])[0]
        self.assertEqual(empty['groundingpi_set'], 1.)
        self.assertEqual(empty['groundingpi_strict'], 1.)
        text = ('<|object_ref_start|>cat<|object_ref_end|><|box_start|>None<|box_end|>, '
                '<|object_ref_start|>dog<|object_ref_end|><|box_start|><0><1><998><999><|box_end|>')
        pred, valid = parser.parse_rex_boxes(text)
        self.assertTrue(valid)
        self.assertEqual(pred[0][0], 'dog')
        self.assertAlmostEqual(pred[0][1][3], 1000.)

    def test_invalid_formats_receive_zero(self):
        prefix = '<|object_ref_start|>cat<|object_ref_end|><|box_start|>'
        invalid = [prefix + '<1><2><3><|box_end|>', prefix + '<1><2><1000><9><|box_end|>',
                   prefix + '<8><2><3><9><|box_end|>', prefix + '<1><2><3><9>',
                   prefix + '<1><2><3><9><|box_end|>,', '{"label":"cat"}']
        for text in invalid:
            score = adapted.compute_score([text], [[]])[0]
            self.assertEqual(score['overall'], 0., text)
            self.assertEqual(score['rex_format_valid'], 0., text)


if __name__ == '__main__':
    unittest.main()
