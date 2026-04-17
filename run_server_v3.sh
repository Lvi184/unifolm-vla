#!/bin/bash
# 启动 UnifoLM-VLA v3 服务器

cd /root/gpufree-data/unifolm-vla
source .venv/bin/activate

echo "🚀 启动 UnifoLM-VLA v3 服务器..."
echo "📍 WebSocket: ws://0.0.0.0:8999"
echo "📍 健康检查: http://0.0.0.0:8999/healthz"
echo ""

python start_policy_server_v3.py
