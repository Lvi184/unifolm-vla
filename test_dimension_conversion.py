#!/usr/bin/env python3
"""
测试维度转换是否正确！
验证我们的21D ↔ 159D/40D转换是否和官方ACoT-VLA一致
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unifolm_openpi_policy import convert_21d_state_to_159d, convert_40d_action_to_21d

def test_official_behavior():
    """测试官方ACoT-VLA的行为"""
    print("=" * 80)
    print("测试1: 验证官方ACoT-VLA的维度转换逻辑")
    print("=" * 80)
    
    # 模拟官方的slice_state_and_action
    def official_slice_state(state_159d):
        """官方ACoT-VLA的状态切片逻辑"""
        if len(state_159d) == 159:
            state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
            return state_159d[state_indices]
        return state_159d
    
    def official_slice_action(action_40d):
        """官方ACoT-VLA的动作切片逻辑"""
        if len(action_40d) == 40:
            # 注意：官方用column_stack是针对2D数组的，我们是1D的
            part1 = action_40d[16:30]  # 14 elements
            part2 = action_40d[0:2]     # 2 elements
            part3 = action_40d[33:38]   # 5 elements
            return np.concatenate([part1, part2, part3])
        return action_40d
    
    print("\n✅ 官方逻辑:")
    print("   - 159D State → 21D State: indices [30-43] + [0-1] + [75-79]")
    print("   - 40D Action → 21D Action: [16-29] + [0-1] + [33-37]")
    
    return official_slice_state, official_slice_action

def test_round_trip_conversion():
    """测试往返转换是否正确"""
    print("\n" + "=" * 80)
    print("测试2: 测试往返转换 (159D → 21D → 159D)")
    print("=" * 80)
    
    # 创建一个测试用的159D状态
    state_159d_original = np.arange(159, dtype=np.float32)
    print(f"\n原始159D状态: {state_159d_original[:10]}... (total {len(state_159d_original)}D)")
    
    # 第一步：官方ACoT-VLA提取21D
    def official_slice(state_159d):
        state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
        return state_159d[state_indices]
    
    state_21d_from_159d = official_slice(state_159d_original)
    print(f"官方提取的21D状态: {state_21d_from_159d}")
    print(f"  (应该是: [30,31,...,43,0,1,75,76,77,78,79] 的值)")
    
    # 第二步：我们的转换把21D填回159D
    state_159d_recovered = convert_21d_state_to_159d(state_21d_from_159d)
    
    # 检查对应位置的值是否正确
    print("\n✅ 验证恢复的值:")
    
    # 检查range(30, 44)
    all_correct = True
    for i, original_idx in enumerate(range(30, 44)):
        expected = state_159d_original[original_idx]
        actual = state_159d_recovered[original_idx]
        if expected == actual:
            print(f"  ✓ position {original_idx}: {actual} (correct)")
        else:
            print(f"  ✗ position {original_idx}: {actual}, expected {expected}")
            all_correct = False
    
    # 检查[0, 1]
    for i, original_idx in enumerate([0, 1]):
        expected = state_159d_original[original_idx]
        actual = state_159d_recovered[original_idx]
        if expected == actual:
            print(f"  ✓ position {original_idx}: {actual} (correct)")
        else:
            print(f"  ✗ position {original_idx}: {actual}, expected {expected}")
            all_correct = False
    
    # 检查range(75, 80)
    for i, original_idx in enumerate(range(75, 80)):
        expected = state_159d_original[original_idx]
        actual = state_159d_recovered[original_idx]
        if expected == actual:
            print(f"  ✓ position {original_idx}: {actual} (correct)")
        else:
            print(f"  ✗ position {original_idx}: {actual}, expected {expected}")
            all_correct = False
    
    if all_correct:
        print("\n✅ 所有位置都正确！往返转换成功！")
    else:
        print("\n❌ 有些位置不正确！")
    
    return all_correct

def test_action_conversion():
    """测试动作转换"""
    print("\n" + "=" * 80)
    print("测试3: 测试动作转换 (40D → 21D)")
    print("=" * 80)
    
    # 创建测试用的40D动作
    action_40d_original = np.arange(40, dtype=np.float32)
    print(f"\n原始40D动作: {action_40d_original}")
    
    # 我们的转换
    action_21d_ours = convert_40d_action_to_21d(action_40d_original)
    
    # 官方逻辑
    def official_action_slice(action_40d):
        part1 = action_40d[16:30]  # 14 elements
        part2 = action_40d[0:2]     # 2 elements
        part3 = action_40d[33:38]   # 5 elements
        return np.concatenate([part1, part2, part3])
    
    action_21d_official = official_action_slice(action_40d_original)
    
    print(f"\n我们转换的21D动作: {action_21d_ours}")
    print(f"官方转换的21D动作: {action_21d_official}")
    
    if np.array_equal(action_21d_ours, action_21d_official):
        print("\n✅ 完全一致！我们的转换和官方ACoT-VLA相同！")
        return True
    else:
        print("\n❌ 不一致！")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("🔍 维度转换正确性验证")
    print("=" * 80)
    
    test_official_behavior()
    
    print("\n" + "=" * 80)
    print("开始详细测试...")
    print("=" * 80)
    
    test1_ok = test_round_trip_conversion()
    test2_ok = test_action_conversion()
    
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    if test1_ok and test2_ok:
        print("\n🎉 所有测试通过！")
        print("✅ 我们的维度转换和官方ACoT-VLA完全一致！")
        print("✅ 我们的接口可以完全替代官方接口！")
        return 0
    else:
        print("\n❌ 有些测试失败了！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
