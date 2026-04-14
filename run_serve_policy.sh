#!/bin/bash

# UnifoLM-VLA OpenPI Policy Server (ACoT-VLA compatible)
# =====================================================
#
# Usage:
#   ./run_serve_policy.sh
#
# Or with custom parameters:
#   source .venv/bin/activate
#   python serve_policy.py \
#     --env G2SIM \
#     --port 8999 \
#     --policy checkpoint \
#     --policy.dir=./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt

# Activate virtual environment
cd /root/gpufree-data/unifolm-vla
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Configuration
CHECKPOINT_PATH="./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"
VLM_PRETRAINED_PATH="/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
UNNORM_KEY="rlds_dataset"
PORT=8999
ENV="G2SIM"

echo "==========================================="
echo " UnifoLM-VLA OpenPI Policy Server"
echo "==========================================="
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "VLM Pretrained: $VLM_PRETRAINED_PATH"
echo "Dataset key: $UNNORM_KEY"
echo "Port: $PORT"
echo "Environment: $ENV"
echo ""

# Run the serve_policy script (ACoT-VLA format)
python serve_policy.py \
  --env "$ENV" \
  --port "$PORT" \
  --vlm_pretrained_path "$VLM_PRETRAINED_PATH" \
  --unnorm_key "$UNNORM_KEY" \
  --policy checkpoint \
  --policy.dir "$CHECKPOINT_PATH"
