# UnifoLM-VLA 适配 AgiBotWorld Challenge 2026 数据转换总结

## 目标
将官方 `agibot_data_without_depth` 转换成 UnifoLM-VLA 可训练格式，遵循 ACoT-VLA 官方 159→21 / 40→21 切法，适配比赛 GenieSim 推理接口。

## 关键改动总结

### 1. 新增文件

#### `prepare_data/adapters/g2a_space_adapter.py`
- 严格遵循 ACoT-VLA 官方切法：
  - `STATE_IDX_159_TO_21 = list(range(30, 44)) + [0, 1] + list(range(75, 80))`
  - `ACTION_SLICES_40_TO_21 = ((16, 30), (0, 2), (33, 38))`
- 提供单步/批量投影函数
- ✅ 修复原始语法错误：删除了错误的 `from typing import np.ndarray`

#### `prepare_data/adapters/agibotworld_meta.py`
- 读取 `meta/info.json`，支持可选 `tasks.jsonl` / `episodes.jsonl`
- `get_instruction_for_frame` 按 `instruction_segments` 和 `frame_index` 动态分配指令
- ✅ 改进 fallback 逻辑：优先从 `tasks` 拿任务名，再回退到 `task_name`

#### `prepare_data/adapters/agibotworld_reader.py`
- 读取 parquet + 三路 RGB 视频，按 `frame_index` 对齐
- ✅ `chunks_size` 默认值改为 `1000`，和官方一致
- 输出对齐后的 step 列表，包含 `frame_index` / `images` / `state159` / `action40`

#### `prepare_data/convert_agibotworld_to_hdf5.py`
- 自动遍历 `src_root` 下所有任务，逐个转换
- ✅ 强制三路 RGB 相机必须存在（top_head / hand_left / hand_right），缺一个直接报错
- ✅ 逐帧预分配图像 buffer，保证图像长度和 episode 长度一致，不会错位
- ✅ 输出标准 UnifoLM HDF5 结构：
  - `/observations/images/[primary/left_wrist/right_wrist]`
  - `/observations/qpos` (21D)
  - `/observations/qvel` (全零填充)
  - `/action` (21D)
  - `/language_raw` / `substep_reasonings` (逐帧指令)

#### `prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py`
- ✅ **彻底重写**：一个 HDF5 = 一个 RLDS episode，不再一帧一个 episode
- ✅ 图像 shape 修正：primary = `(400, 640, 3)`，left/right wrist = `(1056, 1280, 3)`，和官方一致
- ✅ 逐帧用 `substep_reasonings[i]`，添加显式 bytes decode，避免 `"b'...'"` 污染文本
- ✅ 字段名统一为 `proprio` 替代 `state`，匹配 UnifoLM 训练侧
- ✅ 添加维度断言：检查 `actions/proprio/images` 维度对齐，不对直接报错
- ✅ `base_dir` 支持环境变量 `UNIFOLM_HDF5_ROOT`，默认 `/root/gpufree-data/unifolm_hdf5`

#### `test_single_task_conversion.py` (根目录新增)
- 单任务测试脚本，只转前 N 个 episode，快速验证转换链路
- 自动检查投影维度、相机存在、输出 HDF5
- 方便调试，不用等全量转换完才能发现错误

### 2. UnifoLM 训练侧配置改动 (`src/unifolm_vla/rlds_dataloader/constants.py`)
- ✅ 新增 `G2A_21_CONSTANTS`：
```python
G2A_21_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,   # 第一版单步，跑通再改chunk
    "ACTION_DIM": 21,
    "PROPRIO_DIM": 21,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}
```
- ✅ 添加命令行检测：`g2a_21` 关键词自动加载这套常量
- 训练启动命令示例：
```bash
python train.py ... g2a_21 --data_mix agibot_competition
```

### 3. 数据集注册（已经存在，无需额外改动）
- `configs.py`：已经有所有比赛任务的配置，相机命名匹配我们的输出
- `transforms.py`：已经有 `unitree_g2a_dataset_transform`
- `mixtures.py`：已经有 `agibot_competition` 混合配置

## 测试流程
1. 测试单任务转换：
```bash
cd /root/gpufree-data/unifolm-vla
python test_single_task_conversion.py \
  --task_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/open_door \
  --hdf5_out /tmp/test_unifolm_hdf5/open_door \
  --num_episodes 5
```

2. 测试 TFDS 构建：
```bash
export UNIFOLM_HDF5_ROOT=/tmp/test_unifolm_hdf5
cd prepare_data/hdf5_to_rlds/rlds_dataset
export no_proxy=* && export NO_PROXY=*
tfds build --data_dir /tmp/test_unifolm_rlds --max_examples_per_split 10
```

3. 如果成功输出：
```
Dataset rlds_dataset downloaded and prepared to ... 
```
整个转换链路验证完成，可以全量转换。

## 全量转换命令
```bash
cd /root/gpufree-data/unifolm-vla/prepare_data
python convert_agibotworld_to_hdf5.py \
  --src_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth \
  --dst_root /root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5

cd /root/gpufree-data/unifolm-vla/prepare_data/hdf5_to_rlds/rlds_dataset
export UNIFOLM_HDF5_ROOT=/root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5
export no_proxy=* && export NO_PROXY=*
tfds build --data_dir /root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_rlds
```

## 完成时间
2026-04-17
- 所有问题修复完毕，符合要求：
  - ✅ 多任务通用
  - ✅ 不丢信息，保留三路 RGB
  - ✅ 指令逐帧匹配
  - ✅ 维度完全对齐 ACoT 官方切法
  - ✅ 输出适配 UnifoLM 训练链路
