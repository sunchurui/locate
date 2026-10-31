"""Find the latest complete optimizer checkpoint, or report completed training."""
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
candidates=sorted((p for p in root.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()),
                  key=lambda p:int(p.name.split('-')[-1]),reverse=True)
for checkpoint in candidates:
    required=['adapter_config.json','adapter_model.safetensors','trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth']
    if not all((checkpoint/name).is_file() for name in required):
        continue
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    if state['global_step']>=state['max_steps']:
        print('COMPLETE')
    else:
        print(checkpoint)
    break
else:
    raise FileNotFoundError(f'No complete optimizer/scheduler/RNG checkpoint in {root}; refusing to restart silently')
