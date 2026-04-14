#!/bin/bash

# UnifoLM-VLA Policy Server Launch Script (Simple Version)
# Uses the native UnifoLM-VLA deployment interface which is simpler

# Activate virtual environment
cd /root/gpufree-data/unifolm-vla
source .venv/bin/activate

# Configuration
CHECKPOINT_PATH="./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"
VLM_PRETRAINED_PATH="/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
UNNORM_KEY="rlds_dataset"
PORT=8999
HOST="0.0.0.0"

echo "==========================================="
echo " UnifoLM-VLA Policy Server"
echo "==========================================="
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "VLM Pretrained: $VLM_PRETRAINED_PATH"
echo "Dataset key: $UNNORM_KEY"
echo "Port: $PORT"
echo "Host: $HOST"
echo ""

# Run using the native UnifoLM-VLA deployment server
python deployment/model_server/run_real_eval_server.py \
  --ckpt_path "$CHECKPOINT_PATH" \
  --vlm_pretrained_path "$VLM_PRETRAINED_PATH" \
  --unnorm_key "$UNNORM_KEY" \
  --host "$HOST" \
  --port "$PORT" \
  --use_bf16
