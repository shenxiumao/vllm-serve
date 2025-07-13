#!/bin/bash

# 停止所有VLLM模型服务的脚本

echo "正在停止所有VLLM模型服务..."

# 加载模型配置以获取端口信息
source /root/data/vllm-serve/vllm_model_configs.sh

# 停止所有vllm serve进程
pkill -f "vllm serve"

echo "所有VLLM模型服务已停止"