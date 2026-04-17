#!/usr/bin/env python3
"""
分析微调的UnifoLM-VLA模型
"""

import torch
import json
import sys
from pathlib import Path

def load_and_analyze_model(checkpoint_path):
    """加载并分析模型checkpoint"""
    print(f"🔍 加载模型: {checkpoint_path}")
    print(f"📏 文件大小: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")
    print()
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=" * 80)
    print("📊 Checkpoint 结构分析")
    print("=" * 80)
    
    # 检查checkpoint的类型
    print(f"Checkpoint 类型: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint 键: {list(checkpoint.keys())}")
        print()
        
        # 分析每个键
        for key in checkpoint.keys():
            value = checkpoint[key]
            print(f"--- {key} ---")
            print(f"  类型: {type(value)}")
            
            if isinstance(value, dict):
                print(f"  子键数量: {len(value)}")
                print(f"  子键: {list(value.keys())[:10]}...")
                
                # 如果是state_dict，分析参数
                if key in ['model', 'model_state_dict', 'state_dict']:
                    analyze_state_dict(value)
                    
            elif isinstance(value, torch.Tensor):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                print(f"  设备: {value.device}")
                
            elif hasattr(value, '__dict__'):
                print(f"  属性: {[k for k in dir(value) if not k.startswith('_')][:20]}")
            
            print()

def analyze_state_dict(state_dict):
    """分析模型的state_dict"""
    print("\n  📦 参数统计:")
    
    total_params = 0
    trainable_params = 0
    layer_stats = {}
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            
            # 按层分类
            layer_type = name.split('.')[0]
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'count': 0, 'params': 0}
            layer_stats[layer_type]['count'] += 1
            layer_stats[layer_type]['params'] += num_params
    
    print(f"    总参数数量: {total_params:,}")
    print(f"    总参数量: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print()
    
    print("  📊 各层参数统计:")
    for layer, stats in sorted(layer_stats.items(), key=lambda x: -x[1]['params']):
        params_mb = stats['params'] * 4 / 1024**2
        print(f"    {layer:<30} | 参数: {stats['params']:>12,} ({params_mb:>8.2f} MB) | 层数: {stats['count']:>4}")
    
    print()
    print("  🔍 前20个参数名:")
    for i, name in enumerate(list(state_dict.keys())[:20]):
        param = state_dict[name]
        if isinstance(param, torch.Tensor):
            print(f"    {i+1:2d}. {name:<60} shape={param.shape}")

def analyze_config(config_path):
    """分析配置文件"""
    print("\n" + "=" * 80)
    print("⚙️  配置文件分析")
    print("=" * 80)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix == '.yaml':
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"不支持的配置文件格式: {config_path.suffix}")
        return
    
    print(f"\n📁 配置文件: {config_path}")
    print()
    
    # 关键配置信息
    print("🔑 关键配置:")
    
    if 'framework' in config:
        fw = config['framework']
        if 'action_model' in fw:
            am = fw['action_model']
            print(f"  动作维度: {am.get('action_dim', 'N/A')}")
            print(f"  状态维度: {am.get('state_dim', 'N/A')}")
            print(f"  隐藏层大小: {am.get('hidden_size', 'N/A')}")
            print(f"  Diffusion层数: {am.get('diffusion_model_cfg', {}).get('num_layers', 'N/A')}")
    
    if 'trainer' in config:
        tr = config['trainer']
        print(f"  最大训练步数: {tr.get('max_train_steps', 'N/A')}")
        print(f"  学习率: {tr.get('learning_rate', {}).get('base', 'N/A')}")
        print(f"  Batch size: {config.get('datasets', {}).get('vla_data', {}).get('per_device_batch_size', 'N/A')}")
    
    if 'datasets' in config:
        ds = config['datasets'].get('vla_data', {})
        print(f"  数据集路径: {ds.get('data_root_dir', 'N/A')}")
        print(f"  数据混合: {ds.get('data_mix', 'N/A')}")

def main():
    base_path = Path("/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v1")
    
    # 分析模型checkpoint
    checkpoint_path = base_path / "checkpoints/steps_8000_pytorch_model.pt"
    if checkpoint_path.exists():
        load_and_analyze_model(checkpoint_path)
    else:
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
    
    # 分析配置文件
    config_yaml = base_path / "config.yaml"
    if config_yaml.exists():
        analyze_config(config_yaml)
    
    config_json = base_path / "config.json"
    if config_json.exists():
        analyze_config(config_json)

if __name__ == "__main__":
    main()
