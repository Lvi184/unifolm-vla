# 官方 ACoT-VLA 接口 vs 我们的 UnifoLM-VLA 接口对比

## 🔍 关键发现！

### 维度转换真相（官方实现）

**不是32D！而是21D！**

---

## 📊 官方 ACoT-VLA 维度转换

### 1. 状态维度转换 (159D → 21D)

**文件**: `/root/gpufree-data/ACoT-VLA/src/openpi/policies/go2_policy.py`

```python
if len(data["state"]) == 159:
    state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
    data["state"] = data["state"][state_indices]
```

**分解**:
- `list(range(30, 44))` → 14个元素 (30, 31, ..., 43)
- `[0, 1]` → 2个元素
- `list(range(75, 80))` → 5个元素 (75, 76, 77, 78, 79)
- **总计**: 14 + 2 + 5 = **21D**

---

### 2. 动作维度转换 (40D → 21D)

```python
if "actions" in data:
    assert data["actions"].shape[1] == 40
    data["actions"] = np.column_stack((
        data["actions"][:, 16:30],  # 14个元素
        data["actions"][:, 0:2],     # 2个元素
        data["actions"][:, 33:38]    # 5个元素
    ))
```

**分解**:
- `data["actions"][:, 16:30]` → 14个元素 (16-29)
- `data["actions"][:, 0:2]` → 2个元素 (0-1)
- `data["actions"][:, 33:38]` → 5个元素 (33-37)
- **总计**: 14 + 2 + 5 = **21D**

---

### 3. 最终输出

```python
class Go2ACOTOutputs:
    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        return {key: np.asarray(data[key][:, :21]) for key in keys if key in data}
```

**确认**: 输出就是 **21D**！

---

## ❌ 我们之前的错误

### 问题1: 以为是32D
- 我们写了 `convert_32d_state_to_159d()` 和 `convert_40d_action_to_32d()`
- **但实际上应该是21D！**

### 问题2: 维度转换逻辑错误
- 我们之前只是简单的填充/切片
- **没有使用官方精确的索引映射！**

---

## ✅ 正确的转换逻辑

### 从 genie_sim/ACoT-VLA 到 UnifoLM-VLA (推理时)

#### 21D State → 159D State
```python
def convert_21d_state_to_159d(state_21d: np.ndarray) -> np.ndarray:
    """
    将 genie_sim/ACoT-VLA 的21D状态转换为 UnifoLM-VLA 的159D状态
    (反向操作官方的 slice_state_and_action)
    """
    state_159d = np.zeros(159, dtype=np.float32)
    
    # 官方提取的索引:
    # state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
    #              = [30,31,...,43] + [0,1] + [75,76,77,78,79]
    
    # 将21D值填回159D的对应位置
    ptr = 0
    
    # 第一部分: range(30, 44) → 14个元素
    for i in range(30, 44):
        state_159d[i] = state_21d[ptr]
        ptr += 1
    
    # 第二部分: [0, 1] → 2个元素
    state_159d[0] = state_21d[ptr]
    state_159d[1] = state_21d[ptr + 1]
    ptr += 2
    
    # 第三部分: range(75, 80) → 5个元素
    for i in range(75, 80):
        state_159d[i] = state_21d[ptr]
        ptr += 1
    
    return state_159d
```

#### 40D Action → 21D Action
```python
def convert_40d_action_to_21d(action_40d: np.ndarray) -> np.ndarray:
    """
    将 UnifoLM-VLA 的40D动作转换为 genie_sim/ACoT-VLA 的21D动作
    (和官方的 slice_state_and_action 完全一致)
    """
    # 官方逻辑:
    # np.column_stack((
    #     data["actions"][:, 16:30],  # 14个元素
    #     data["actions"][:, 0:2],     # 2个元素
    #     data["actions"][:, 33:38]    # 5个元素
    # ))
    
    part1 = action_40d[16:30]  # 14个元素
    part2 = action_40d[0:2]     # 2个元素
    part3 = action_40d[33:38]   # 5个元素
    
    action_21d = np.concatenate([part1, part2, part3])
    return action_21d
```

---

## 📷 图像处理对比

### 官方 ACoT-VLA
```python
rename_map = {
    "top_head": "base_0_rgb",
    "hand_left": "left_wrist_0_rgb", 
    "hand_right": "right_wrist_0_rgb"
}
```
- 重命名摄像头键，但内容不变
- 确保uint8格式和[H,W,C]格式

### 我们的 UnifoLM-VLA
- 类似的图像处理流程
- 额外的resize到224x224
- 可选的center crop

---

## 🎯 总结

| 方面 | 官方 ACoT-VLA | 我们的 UnifoLM-VLA (之前) | 我们的 UnifoLM-VLA (修正后) |
|------|----------------|--------------------------|-----------------------------|
| 状态维度 | 21D | 错误的32D | ✅ 正确的21D |
| 动作维度 | 21D | 错误的32D | ✅ 正确的21D |
| 状态索引映射 | 精确索引 | 简单填充 | ✅ 精确索引 |
| 动作索引映射 | 精确索引 | 简单切片 | ✅ 精确索引 |
| 摄像头处理 | 重命名键 | 类似 | ✅ 保持一致 |

---

## 💡 结论

**我们的接口可以完全替代官方的，只要修正维度转换逻辑！**

修正后：
- ✅ 使用完全相同的21D维度
- ✅ 使用完全相同的索引映射
- ✅ 图像处理保持兼容
- ✅ 可以无缝替换官方接口！
