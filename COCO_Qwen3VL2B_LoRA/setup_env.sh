#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASEPY="${BASE_PYTHON:-python3}"
if [[ ! -x "$ROOT/venv/bin/python" ]]; then
  "$BASEPY" -m venv --system-site-packages --without-pip "$ROOT/venv"
fi
source "$ROOT/venv/bin/activate"
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
  numpy==1.26.4 pyarrow==19.0.1 ms-swift==4.6.0 qwen-vl-utils==0.0.14 \
  pycocotools==2.0.11 modelscope
python -c "import torch, transformers, swift; from qwen_vl_utils import process_vision_info; from pycocotools.cocoeval import COCOeval; print('torch', torch.__version__, 'cuda-build', torch.version.cuda); print('transformers', transformers.__version__); print('swift', swift.__path__)"
