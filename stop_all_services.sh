#!/bin/bash

# 一键停止所有VLLM模型服务和API服务器的脚本

echo "===== 停止所有VLLM模型服务和API服务器 ====="

# 首先停止API服务器
echo "正在停止VLLM API服务器..."
bash /root/data/vllm-serve/stop_api_server.sh

# 然后停止所有模型服务
echo "正在停止所有VLLM模型服务..."
bash /root/data/vllm-serve/stop_all_models.sh

echo "===== 所有服务已停止 ====="