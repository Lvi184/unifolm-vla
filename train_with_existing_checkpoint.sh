#!/bin/bash
# Training script that tries to use existing checkpoint as pretrained initialization

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  

# Try to use offline mode for transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# model 
# vlm model
Framework_name=unifolm_vla
# Try using the existing VLA-Base as starting point
base_vlm=/root/gpufree-data/unifolm-weights/UnifoLM-VLA-Base
model_type=qwen2_5_vl
freeze_module_list=''
window_size=1
# dataset
# vla dataset
oxe_data_root=/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth
data_mix=agibot_competition

# run save path
run_root_dir=/root/gpufree-data/unifolm-vla/results
run_id=unifolm_vla_agibot_v2

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
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 10000 \
  --trainer.shuffle_buffer_size 10000 \
  --trainer.save_interval 2000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 500 \
  --trainer.learning_rate.base 1e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
