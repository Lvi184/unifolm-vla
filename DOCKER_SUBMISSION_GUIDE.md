# UnifoLM-VLA Docker 提交流程

## 📋 需要下载到 Windows 的文件

### 1. **UnifoLM-VLA 项目目录**
```
/root/gpufree-data/unifolm-vla/
├── src/                      # 源代码（完整）
├── unifolm_openpi_policy.py  # openpi 接口适配器
├── start_policy_server_final_v5.py  # WebSocket 服务器
├── pyproject.toml            # 依赖配置
└── 其他必要文件
```

### 2. **模型权重**
```
/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt
```

### 3. **VLM 权重**（可选，如果 Docker 里没有预加载）
```
/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base/
```

---

## 🐳 Dockerfile 模板

```dockerfile
# Dockerfile for UnifoLM-VLA
# Based on ACoT-VLA's Docker setup

FROM registry.agibot.com/genie-sim/openpi_server:latest

# Set working directory
WORKDIR /app

# Copy UnifoLM-VLA project
COPY . .

# Install dependencies (if not already in base image)
# RUN pip install -e .
# Or if you have a requirements.txt:
# RUN pip install -r requirements.txt

# Expose port
EXPOSE 8999

# Set environment variables (similar to ACoT-VLA)
ENV TF_NUM_INTRAOP_THREADS=16
ENV CUDA_VISIBLE_DEVICES=0
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV XLA_FLAGS="--xla_gpu_autotune_level=0"
ENV PYTHONPATH=/app:/app/src
ENV GIT_LFS_SKIP_SMUDGE=1

# Command to start server
# You need to create a server.sh script similar to ACoT-VLA's
CMD ["/bin/bash", "-c", "./scripts/server.sh 0 8999"]
```

---

## 🚀 server.sh 脚本模板

在 `/root/gpufree-data/unifolm-vla/scripts/server.sh` 中：

```bash
#!/bin/bash
# UnifoLM-VLA server startup script
# Similar to ACoT-VLA's server.sh

cart_num=${1}
port=${2}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/app:/app/src

# Start UnifoLM-VLA WebSocket server
python start_policy_server_final_v5.py
```

---

## 📝 完整提交流程（和你之前的类似）

### 1. **启动临时容器**
```bash
docker run -d --name temp-unifolm-server \
  sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest \
  sleep infinity
```

### 2. **清空旧检查点**
```bash
docker exec temp-unifolm-server rm -rf /app/checkpoints/*
```

### 3. **创建目录结构**
```bash
docker exec temp-unifolm-server mkdir -p /app/checkpoints/unifolm_vla/
```

### 4. **拷贝 UnifoLM-VLA 项目文件**
```bash
# 从 Windows 路径拷贝（WSL2 自动识别）
docker cp N:\BaiduNetdiskDownload\unifolm-vla temp-unifolm-server:/app/
```

### 5. **拷贝模型权重**
```bash
docker cp N:\BaiduNetdiskDownload\steps_8000_pytorch_model.pt \
  temp-unifolm-server:/app/checkpoints/unifolm_vla/
```

### 6. **拷贝 VLM 权重（如果需要）**
```bash
docker cp N:\BaiduNetdiskDownload\UnifoLM-VLM-Base \
  temp-unifolm-server:/app/
```

### 7. **打包成新镜像**
```bash
docker commit \
  --change='CMD ["/bin/bash", "./scripts/server.sh", "0", "8999"]' \
  temp-unifolm-server \
  sim-icra-registry.cn-beijing.cr.aliyuncs.com/arobot/unifolm-vla:v1
```

### 8. **清理临时容器**
```bash
docker rm -f temp-unifolm-server
```

### 9. **推送镜像（可选）**
```bash
docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/arobot/unifolm-vla:v1
```

---

## ⚠️ 注意事项

1. **确保路径正确**：根据你 Windows 上的实际路径调整
2. **检查依赖**：确保 Docker 镜像里已经有所有需要的 Python 包
3. **端口映射**：确保 `--network_mode=host` 或正确映射端口

---

## ✅ 验证

在本地测试一下 Docker 镜像是否能正常工作：

```bash
docker run -it --gpus all --network=host \
  sim-icra-registry.cn-beijing.cr.aliyuncs.com/arobot/unifolm-vla:v1 \
  /bin/bash
```

然后在容器里测试：
```bash
python start_policy_server_final_v5.py
```

---

**祝你提交成功！** 🎉
