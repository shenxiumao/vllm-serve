# VLLM API服务器

## 项目概述

这是一个基于FastAPI构建的高性能API服务器，为多个VLLM大语言模型提供统一的接口。本服务器具有以下特点：

- 支持API密钥认证机制
- 请求自动转发到不同端口的VLLM模型服务
- 优化的GPU和CPU资源分配
- 完整的日志记录和监控
- 简单易用的启停脚本
- 支持流式输出和非流式输出
- 自动生成和管理API密钥

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 服务管理

#### 启动服务

**一键启动所有服务**

```bash
bash start_all_services.sh
```

**单独启动API服务器**

```bash
bash start_api_server.sh
```

**单独启动模型服务**

```bash
bash start_all_models.sh
```

#### 停止服务

**一键停止所有服务**

```bash
bash stop_all_services.sh
```

**单独停止API服务器**

```bash
bash stop_api_server.sh
```

**单独停止模型服务**

```bash
bash stop_all_models.sh
```

## API文档

### 认证机制

API服务器使用API密钥进行认证，可通过以下两种方式提供密钥：

- 请求头：`Authorization: Bearer <api_key>`
- 查询参数：`?api_key=<api_key>`

> 首次启动服务器时，系统会自动生成API密钥并保存到`api_keys.json`文件中。

### 接口列表

#### 获取模型列表

```http
GET /v1/models
```

返回所有可用模型的列表及其路径信息。

#### 获取模型详情

```http
GET /v1/models/{model_id}
```

返回指定模型的详细信息。

#### 聊天完成接口

```http
POST /v1/chat/completions
```

**请求示例：**

```json
{
  "model": "DeepSeek-R1-0528-Qwen3-14B",
  "messages": [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好，请介绍一下自己。"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

#### 健康检查

```http
GET /health
```

返回服务器健康状态、可用模型列表和路径信息。

#### 管理API密钥

```http
GET /admin/keys
```

列出所有API密钥信息（需要管理员API密钥权限）。

## 系统日志

- API服务器日志：`vllm_api_server.log`
- 模型服务日志：
  - `deepseek_qwen3_vllm.log`
  - `qwen3_14b_vllm.log`
  - `deepseek_distill_vllm.log`
  - `qwen_vllm.log`

## 系统配置

### API服务器配置

API服务器配置可在`vllm_api_server.py`文件中修改，包括：

- 服务端口：1024
- 工作线程数量：32
- 并发限制：100
- 连接保持时间：300秒
- HTTP解析器：httptools
- 速率限制：每分钟20个请求

### VLLM模型配置

所有VLLM模型的配置参数集中在`vllm_model_configs.sh`文件中：

- 模型路径
- 服务端口
- 张量并行大小：4
- GPU内存利用率：14B模型为0.325，7B模型为0.155
- 最大模型长度：32768
- 最大序列数：8
- 并行加载工作线程：64

## 性能优化

### GPU和CPU资源分配

系统采用以下GPU和CPU线程绑定策略优化性能：

- GPU0 → CPU线程 0-15
- GPU1 → CPU线程 16-31
- GPU2 → CPU线程 48-63
- GPU3 → CPU线程 64-79

推理时共占用64个CPU线程，确保资源高效利用。所有模型使用张量并行大小为4，充分利用多GPU资源。

### 环境变量优化

#### 线程数设置

```
OMP_NUM_THREADS=48
MKL_NUM_THREADS=48
NUMEXPR_NUM_THREADS=48
NUMEXPR_MAX_THREADS=48
TOKENIZERS_PARALLELISM=true
CUDA_DEVICE_MAX_CONNECTIONS=16
```

#### PyTorch和VLLM特定参数

```
TORCH_CPU_CORES=64
VLLM_CPU_OMP_THREADS_BIND=64
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

#### 线程亲和性和调度策略

```
KMP_AFFINITY="granularity=fine,compact,1,0"
KMP_BLOCKTIME=0
OMP_SCHEDULE="dynamic"
```

### 其他性能优化措施

- 使用numactl进行CPU线程与GPU的绑定
- 模型启动时间间隔设置为15秒，避免资源竞争
- 使用enforce-eager模式提高推理稳定性
- 优化线程亲和性和动态调度策略，提高线程利用效率

## 技术架构

### 组件结构

- **API服务器**：处理客户端请求，进行认证和请求转发
- **模型服务**：每个模型独立运行在不同端口
- **配置管理**：集中管理所有模型和服务器配置
- **认证系统**：API密钥生成和验证
- **日志系统**：记录所有请求和系统状态

### 请求流程

1. 客户端发送请求到API服务器（端口1024）
2. API服务器验证API密钥和权限
3. 根据请求中的模型名称，转发到对应的模型服务
4. 模型服务处理请求并返回结果
5. API服务器将结果返回给客户端

## 依赖项

主要依赖包括：

- FastAPI：Web框架
- Uvicorn：ASGI服务器
- HTTPX：HTTP客户端
- VLLM：高性能推理引擎
- PyTorch：深度学习框架
- Pydantic：数据验证

完整依赖列表请参见`requirements.txt`文件。