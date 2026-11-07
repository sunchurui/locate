"""Small CPU integration check; uses the project's tokenizer, no model or training."""
from pathlib import Path
import sys
from transformers import AddedToken, AutoTokenizer
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'references/Rex-Omni/finetuning'))
from verl.trainer.object_credit import credit_for_response, apply_object_credit

tok = AutoTokenizer.from_pretrained(root / 'model', local_files_only=True)
tok.add_tokens([AddedToken(f'<{i}>', normalized=False, special=True) for i in range(1000)], special_tokens=True)
text = ('<|object_ref_start|>cat<|object_ref_end|><|box_start|>'
        '<100><100><300><300>,<110><110><290><290><|box_end|><|im_end|>')
ids = tok.encode(text, add_special_tokens=False)
truth = [{'label': 'cat', 'bbox_2d': [100 * 1000 / 999, 100 * 1000 / 999,
                                       300 * 1000 / 999, 300 * 1000 / 999]},
         {'label': 'dog', 'bbox_2d': [600, 600, 800, 800]}]
local, stats = credit_for_response(ids, truth, tok, complete_gt=True, allowed_labels={'cat', 'dog'})
assert stats['matched'] == 1 and stats['unmatched'] == 1
assert stats['absent_category_missed'] == 1
assert sum(value > 0 for value in local.values()) == 4
assert sum(value == -1. for value in local.values()) == 4
assert local[len(ids)-1] == -0.5

class Data:
    def __init__(self):
        self.batch = {'responses': torch.tensor([ids]), 'response_mask': torch.ones((1, len(ids)), dtype=torch.long),
                      'advantages': torch.full((1, len(ids)), 0.5), 'returns': torch.full((1, len(ids)), 0.5)}
        self.non_tensor_batch = {'ground_truth': [truth]}
    def __len__(self):
        return 1

data = Data()
apply_object_credit(data, tok, .1, complete_gt=True, allowed_labels={'cat', 'dog'})
assert torch.allclose(data.batch['returns'], torch.full((1, len(ids)), 0.5))
assert abs(float(data.batch['advantages'][0, ids.index(tok.convert_tokens_to_ids('<100>'))]) - 0.6) < 1e-5
print('PASS: real tokenizer, matched box, extra box, missing category EOS, advantage and unchanged returns')
