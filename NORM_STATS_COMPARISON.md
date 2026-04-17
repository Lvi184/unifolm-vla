# Norm Stats 对比分析：ACoT-VLA vs UnifoLM-VLA

## 📊 关键区别

### 1. 维度差异

| 项目 | 状态维度 | 动作维度 |
|------|---------|---------|
| **ACoT-VLA (norm_stats.json)** | 32D | 32D |
| **ACoT-VLA (实际代码使用)** | 21D | 21D |
| **UnifoLM-VLA (训练)** | 159D | 40D |
| **UnifoLM-VLA (推理)** | 21D (转换后) | 21D (转换后) |

---

### 2. 归一化统计数据来源

#### ACoT-VLA
- **文件**: `/root/gpufree-data/ACoT-VLA/checkpoints/baseline/30000/assets/norm_stats.json`
- **特点**: 
  - 32D state / 32D actions
  - 包含mean, std, q01, q99
  - 很多位置是0（可能是mask掉的维度）

#### UnifoLM-VLA
- **文件**: `/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/dataset_statistics.json`
- **特点**:
  - 159D proprio (state) / 40D action
  - 包含mean, std, max, min, q01, q99
  - 从训练数据集中直接计算得到

---

### 3. 维度转换的影响

#### 重要发现！🔍

**UnifoLM-VLA和ACoT-VLA使用的是完全不同的归一化统计数据！**

但这**不会影响**我们的维度转换逻辑，原因如下：

1. **维度索引映射是独立于归一化的**
   - 我们只是把21D的值填回到159D的正确位置
   - 归一化是在填回之后才进行的
   - UnifoLM-VLA使用自己的norm_stats对159D进行归一化

2. **推理时的流程**
   ```
   genie_sim (21D state)
       ↓
   [我们的转换] 填回到159D的正确位置
       ↓
   [UnifoLM-VLA] 使用自己的norm_stats归一化159D
       ↓
   [模型推理] 输出40D action
       ↓
   [UnifoLM-VLA] 使用自己的norm_stats反归一化40D
       ↓
   [我们的转换] 从40D提取21D（和ACoT-VLA一样的索引）
       ↓
   genie_sim (21D action)
   ```

3. **关键点**
   - ✅ UnifoLM-VLA在训练时已经学习了159D/40D的分布
   - ✅ UnifoLM-VLA有自己的norm_stats，不需要ACoT-VLA的
   - ✅ 我们只是在接口层面做维度匹配，不改变归一化逻辑

---

### 4. 实际验证

让我们看看两个项目的实际维度使用：

#### ACoT-VLA的go2_policy.py
```python
# 从159D提取21D
if len(data["state"]) == 159:
    state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
    data["state"] = data["state"][state_indices]

# 从40D提取21D
if "actions" in data:
    assert data["actions"].shape[1] == 40
    data["actions"] = np.column_stack((
        data["actions"][:, 16:30], 
        data["actions"][:, 0:2], 
        data["actions"][:, 33:38]
    ))
```

#### UnifoLM-VLA的constants.py（G2A设置）
```python
ACTION_DIM = 40
PROPRIO_DIM = 159
```

---

### 5. 总结

| 问题 | 答案 |
|------|------|
| **UnifoLM-VLA需要ACoT-VLA的norm_stats.json吗？** | ❌ 不需要！UnifoLM-VLA有自己的dataset_statistics.json |
| **维度处理不同会有问题吗？** | ❌ 不会！我们只是在接口层做维度匹配，不影响归一化 |
| **我们的接口能替代官方的吗？** | ✅ 可以！只要维度索引映射正确就行 |
| **归一化会影响最终动作吗？** | ❌ 不会！每个模型用自己训练时的norm_stats |

---

## 🎯 结论

**你的观察非常准确！** 两个项目确实：
1. 使用不同的norm_stats文件
2. 训练时使用不同的维度（159D/40D vs 32D/32D）
3. 但推理时都使用相同的21D接口

**但这不影响我们的接口替代方案！** 因为：
- ✅ UnifoLM-VLA已经用自己的norm_stats训练好了
- ✅ 我们只是在接口层面做维度转换，不改变模型内部的归一化
- ✅ 只要维度索引映射和ACoT-VLA一致，就能输出正确的21D动作

**所以我们的方案是完全可行的！** 🚀
