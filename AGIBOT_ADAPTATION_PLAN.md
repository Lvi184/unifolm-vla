# UnifoLM-VLA 适配 AGIBOT 比赛计划

## 当前状态
- ✅ UnifoLM-VLA 仓库已克隆到 `/root/gpufree-data/unifolm-vla/`
- ✅ 已读取中文 README，了解项目结构
- ✅ 已分析比赛数据集结构
- ❌ HuggingFace 权重下载失败（网络问题）
- ✅ 在 configs.py 中添加了 G2A 的 StateEncoding 和 ActionEncoding
- ✅ 在 constants.py 中添加了 G2A 的常量配置
- ✅ 更新了 detect_robot_platform 函数，添加了对 G2A 的支持
- ✅ 更新了常量设置部分，添加了对 G2A 的支持
- ✅ 添加了 15 个比赛数据集的配置到 configs.py
- ✅ 添加了 15 个比赛数据集的 transform 配置到 transforms.py
- ✅ 添加了比赛数据集的混合配置到 mixtures.py
- ✅ 创建了新的训练配置文件 `unifolm_vla_agibot_train.yaml`
- ✅ 创建了新的训练脚本 `run_unifolm_vla_agibot_train.sh`

---

## 比赛数据集关键信息
- **机器人类型**：G2A
- **状态维度**：159
- **动作维度**：40
- **摄像头**：
  - top_head (400x640) → observation.images.top_head
  - hand_left (1056x1280) → observation.images.hand_left
  - hand_right (1056x1280) → observation.images.hand_right
- **数据集格式**：LeRobot V2.1
- **数据集位置**：`/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/`

---

## 比赛数据集列表（15个子数据集）
1. `clean_the_desktop_addition`
2. `clean_the_desktop_part_1`
3. `clean_the_desktop_part_2`
4. `hold_pot`
5. `open_door`
6. `place_block_into_box`
7. `pour_workpiece`
8. `scoop_popcorn`
9. `scoop_popcorn_part_2`
10. `sorting_packages_part_1`
11. `sorting_packages_part_2`
12. `sorting_packages_part_3`
13. `stock_and_straighten_shelf`
14. `stock_and_straighten_shelf_part_2`
15. `take_wrong_item_shelf`

---

## 需要完成的适配工作

### 1. 添加比赛数据集配置到 configs.py
**文件位置**：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py`

需要添加 15 个比赛数据集的配置，参考 G1 数据集的配置格式。

**关键配置**：
- `image_obs_keys`: 映射到比赛数据集的摄像头键名
- `state_obs_keys`: ["state"]
- `state_encoding`: 需要为 G2A 定义新的编码方式
- `action_encoding`: 需要为 G2A 定义新的编码方式

### 2. 添加比赛数据集到 transforms.py
**文件位置**：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py`

需要添加 15 个比赛数据集的 transform 配置。

### 3. 添加比赛数据集到 mixtures.py
**文件位置**：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py`

需要添加比赛数据集的混合配置。

### 4. 定义 G2A 的 StateEncoding 和 ActionEncoding
**文件位置**：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py`

需要为 G2A 机器人定义新的状态和动作编码方式。

### 5. 下载 UnifoLM-VLA-Base 预训练权重
**目标位置**：`/root/gpufree-data/unifolm-weights/Unifolm-VLA-Base/`

由于 HuggingFace 连接失败，可能需要：
- 手动下载
- 使用代理
- 或者稍后重试

### 6. 修改训练配置
**文件位置**：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml`

需要修改：
- `base_vlm`: 指向下载的权重路径
- `data_mix`: 指向比赛数据集混合
- `action_dim`: 40（比赛动作维度）
- `state_dim`: 159（比赛状态维度）

### 7. 修改训练脚本
**文件位置**：`/root/gpufree-data/unifolm-vla/scripts/run_scripts/run_unifolm_vla_train.sh`

需要修改：
- `base_vlm`: 指向下载的权重路径
- `oxe_data_root`: 指向比赛数据集路径
- `data_mix`: 指向比赛数据集混合
- 其他训练参数

---

## 优先级

### 高优先级
1. ✅ 添加比赛数据集配置到 configs.py
2. ✅ 定义 G2A 的 StateEncoding 和 ActionEncoding
3. ✅ 添加比赛数据集到 transforms.py
4. ✅ 添加比赛数据集到 mixtures.py

### 中优先级
5. 下载 UnifoLM-VLA-Base 预训练权重（你来做）
6. ✅ 修改训练配置（创建了新的配置文件）
7. ✅ 修改训练脚本（创建了新的训练脚本）

### 低优先级
8. 开始训练和测试

---

## 下一步行动

1. **立即开始**：添加比赛数据集配置
2. **同时进行**：尝试其他方式下载权重
3. **完成后**：开始训练和测试

---

## 相关文件位置

- UnifoLM-VLA 仓库：`/root/gpufree-data/unifolm-vla/`
- 比赛数据集：`/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/`
- 数据集配置：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py`
- 数据集 transforms：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py`
- 数据集 mixtures：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py`
- 训练配置：`/root/gpufree-data/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml`
- 训练脚本：`/root/gpufree-data/unifolm-vla/scripts/run_scripts/run_unifolm_vla_train.sh`
