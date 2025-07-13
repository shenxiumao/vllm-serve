#!/bin/bash

# 停止VLLM API服务器的脚本

echo "正在停止VLLM API服务器..."

# 查找并终止API服务器进程
API_PID=$(ps -ef | grep "python /root/data/vllm-serve/vllm_api_server.py" | grep -v grep | awk '{print $2}')

if [ -z "$API_PID" ]; then
    echo "未找到运行中的VLLM API服务器进程"
else
    echo "找到VLLM API服务器进程 (PID: $API_PID)，正在终止..."
    kill -15 $API_PID
    sleep 2
    
    # 检查进程是否已终止
    if ps -p $API_PID > /dev/null; then
        echo "进程未响应SIGTERM信号，正在强制终止..."
        kill -9 $API_PID
    fi
    
    echo "VLLM API服务器已停止"
fi