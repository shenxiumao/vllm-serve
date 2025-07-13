#!/bin/bash

# 一键启动所有VLLM模型服务和API服务器的脚本

echo "===== 启动所有VLLM模型服务和API服务器 ====="

# 首先启动所有模型服务
echo "正在启动所有VLLM模型服务..."
bash /root/data/vllm-serve/start_all_models.sh

# 等待100秒，确保所有模型服务都已启动
echo "等待60秒，确保所有模型服务都已启动..."
sleep 60

# 启动API服务器
echo "正在启动VLLM API服务器..."
bash /root/data/vllm-serve/start_api_server.sh

echo "===== 所有服务已启动 ====="
echo "API服务器运行在端口1024上"
echo "模型服务运行在以下端口："
echo "- DeepSeek-R1-0528-Qwen3-14B: 端口8000"
echo "- Qwen3-14B: 端口8001"
echo "- DeepSeek-R1-Distill-Qwen-7B: 端口8002"
echo "- Qwen2.5-7B-Instruct: 端口8003"

echo "使用以下命令查看API服务器日志："
echo "tail -f /root/data/vllm-serve/vllm_api_server.log"

echo "使用以下命令查看模型服务日志："
echo "tail -f /root/data/vllm-serve/deepseek_qwen3_vllm.log"
echo "tail -f /root/data/vllm-serve/qwen3_14b_vllm.log"
echo "tail -f /root/data/vllm-serve/deepseek_distill_vllm.log"
echo "tail -f /root/data/vllm-serve/qwen_vllm.log"