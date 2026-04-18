#!/usr/bin/env bash
set -e

# Minimal smoke test for agibot_g2a_21
# Usage:
#   bash train_agibot_g2a_min.sh

cd /root/gpufree-data/unifolm-vla
source /root/gpufree-data/unifolm-vla/.venv/bin/activate

# Environment: disable cloud checks
export TFDS_DISABLE_GCS=1
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=2

# Change this to your actual base VLM path
BASE_VLM=/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base
# Output root
RUN_ROOT=/root/gpufree-data/unifolm-runs
RUN_ID=agibot_g2a_21_smoke

mkdir -p ${RUN_ROOT}/${RUN_ID}

accelerate launch \
 --num_processes 1 \
 src/unifolm_vla/training/train_unifolm_vla.py \
 --config_yaml /root/gpufree-data/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_agibot_train.yaml \
 --framework.framework_py unifolm_vla \
 --framework.qwenvl.base_vlm ${BASE_VLM} \
 --framework.qwenvl.model_type qwen2_5_vl \
 --datasets.vla_data.data_root_dir /root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_rlds \
 --datasets.vla_data.data_mix rlds_dataset \
 --datasets.vla_data.window_size 1 \
 --datasets.vla_data.per_device_batch_size 1 \
 --trainer.freeze_modules "" \
 --trainer.max_train_steps 200 \
 --trainer.shuffle_buffer_size 100 \
 --trainer.save_interval 100 \
 --trainer.eval_interval 100000 \
 --trainer.logging_frequency 10 \
 --trainer.use_wrist_image True \
 --trainer.use_proprio True \
 --trainer.learning_rate.base 1e-5 \
 --run_root_dir ${RUN_ROOT} \
 --run_id ${RUN_ID}
