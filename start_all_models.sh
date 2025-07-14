#!/bin/bash

# 启动所有VLLM模型服务的脚本
# 实现GPU与CPU线程绑定
# GPU0绑定到（0～15），GPU1绑定到（16～31），GPU2绑定到（48～63），GPU3绑定到（64-79）

# 加载模型配置
source /root/data/vllm-serve/vllm_model_configs.sh

# 启动VLLM服务的脚本，增加了工作线程数量和GPU并行时的CPU线程数

# 设置环境变量以增加CPU线程数
export OMP_NUM_THREADS=48       # OpenMP线程数
export MKL_NUM_THREADS=48       # MKL线程数
export NUMEXPR_NUM_THREADS=48   # NumExpr线程数
export NUMEXPR_MAX_THREADS=48   # NumExpr最大线程数
export TOKENIZERS_PARALLELISM=true  # Huggingface tokenizers并行化

# 设置每个GPU的CUDA线程数为16，4卡共64线程
export CUDA_DEVICE_MAX_CONNECTIONS=16  # 每个GPU设备的最大连接数

export TORCH_CPU_CORES=64      # 告诉PyTorch使用64个CPU核心
export VLLM_CPU_OMP_THREADS_BIND=64  # VLLM特定参数，强制使用64个线程

export KMP_AFFINITY="granularity=fine,compact,1,0"  # 线程亲和性设置
export KMP_BLOCKTIME=0         # 线程阻塞时间，0表示立即释放
export OMP_SCHEDULE="dynamic"   # 动态调度策略

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 设置日志目录
LOG_DIR="/root/data/vllm-serve"

echo "===== 启动所有VLLM模型服务 ====="

# 启动模型1: DeepSeek-R1-0528-Qwen3-14B
echo "正在启动 DeepSeek-R1-0528-Qwen3-14B 模型服务..."
numactl --physcpubind=0-15,16-31,48-63,64-79 vllm serve /root/data/model/zjydiary/DeepSeek-R1-0528-Qwen3-14B \
    --served-model-name DeepSeek-R1-0528-Qwen3-14B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enforce-eager \
    --max-parallel-loading-workers 64 \
    --gpu-memory-utilization 0.325 \
    --port 8000 > $LOG_DIR/deepseek_qwen3_vllm.log 2>&1 &
echo "DeepSeek-R1-0528-Qwen3-14B 模型服务已在后台启动，端口: $DEEPSEEK_QWEN3_PORT"

# 等待15秒，确保第一个模型启动完成
sleep 15

# 启动模型2: Qwen3-14B
echo "正在启动 Qwen3-14B 模型服务..."
numactl --physcpubind=0-15,16-31,48-63,64-79 vllm serve /root/data/model/Qwen/Qwen3-14B \
    --served-model-name Qwen3-14B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enforce-eager \
    --max-parallel-loading-workers 64 \
    --gpu-memory-utilization 0.325 \
    --port 8001 > $LOG_DIR/qwen3_14b_vllm.log 2>&1 &
echo "Qwen3-14B 模型服务已在后台启动，端口: $QWEN3_14B_PORT"

# 等待15秒，确保第二个模型启动完成
sleep 15

# 启动模型3: DeepSeek-R1-Distill-Qwen-7B
echo "正在启动 DeepSeek-R1-Distill-Qwen-7B 模型服务..."
numactl --physcpubind=0-15,16-31,48-63,64-79 vllm serve /root/data/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enforce-eager \
    --max-parallel-loading-workers 64 \
    --gpu-memory-utilization 0.155 \
    --port 8002 > $LOG_DIR/deepseek_distill_vllm.log 2>&1 &
echo "DeepSeek-R1-Distill-Qwen-7B 模型服务已在后台启动，端口: $DEEPSEEK_DISTILL_PORT"

# 等待15秒，确保第三个模型启动完成
sleep 15

# 启动模型4: Qwen2.5-7B-Instruct
echo "正在启动 Qwen2.5-7B-Instruct 模型服务..."
numactl --physcpubind=0-15,16-31,48-63,64-79 vllm serve /root/data/model/Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enforce-eager \
    --max-parallel-loading-workers 64 \
    --gpu-memory-utilization 0.155 \
    --port 8003 > $LOG_DIR/qwen_vllm.log 2>&1 &
echo "Qwen2.5-7B-Instruct 模型服务已在后台启动，端口: $QWEN_7B_PORT"

echo "===== 所有VLLM模型服务已启动 ====="
echo "模型服务运行在以下端口："
echo "- DeepSeek-R1-0528-Qwen3-14B: 端口$DEEPSEEK_QWEN3_PORT"
echo "- Qwen3-14B: 端口$QWEN3_14B_PORT"
echo "- DeepSeek-R1-Distill-Qwen-7B: 端口$DEEPSEEK_DISTILL_PORT"
echo "- Qwen2.5-7B-Instruct: 端口$QWEN_7B_PORT"

echo "使用以下命令查看模型服务日志："
echo "tail -f $LOG_DIR/deepseek_qwen3_vllm.log"
echo "tail -f $LOG_DIR/qwen3_14b_vllm.log"
echo "tail -f $LOG_DIR/deepseek_distill_vllm.log"
echo "tail -f $LOG_DIR/qwen_vllm.log"