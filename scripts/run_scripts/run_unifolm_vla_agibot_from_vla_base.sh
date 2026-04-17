#!/bin/bash
# Training script for UnifoLM-VLA on AGIBOT 2026 Competition Dataset
# Fine-tuning from UnifoLM-VLA-Base checkpoint

# Activate virtual environment
source /root/gpufree-data/unifolm-vla/.venv/bin/activate

# Simplified NCCL settings for single GPU training
export NCCL_TIMEOUT=1000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

# model 
# vlm model - we still need VLM base for the QWen model
Framework_name=unifolm_vla
base_vlm=/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base
model_type=qwen2_5_vl
freeze_module_list=''
window_size=1
# dataset
# vla dataset
oxe_data_root=/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_rlds
data_mix=agibot_competition

# run save path
run_root_dir=/root/gpufree-data/unifolm-vla/results
run_id=unifolm_vla_agibot_v3_finetune_from_vla_base

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# Note: You may need to adjust num_processes based on your GPU setup
# For now, let's use 1 GPU for testing, you can increase later
accelerate launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_agibot_train.yaml \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.model_type ${model_type} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.window_size ${window_size} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 10000 \
  --trainer.shuffle_buffer_size 10000 \
  --trainer.save_interval 2000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 10000 \
  --trainer.learning_rate.base 1e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
