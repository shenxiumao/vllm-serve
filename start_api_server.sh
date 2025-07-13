#!/bin/bash

# 启动VLLM API服务器的脚本

echo "正在启动VLLM API服务器..."

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:/root/data/vllm-serve"

# 启动API服务器
nohup python /root/data/vllm-serve/vllm_api_server.py > /root/data/vllm-serve/vllm_api_server.log 2>&1 &

echo "VLLM API服务器已在后台启动，日志文件：/root/data/vllm-serve/vllm_api_server.log"
echo "服务器运行在端口1024上"
echo "使用以下命令查看日志："
echo "tail -f /root/data/vllm-serve/vllm_api_server.log"