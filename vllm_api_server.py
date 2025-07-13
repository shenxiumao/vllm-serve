from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field
import httpx
import logging
import uvicorn
import json
import asyncio
import os
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from collections import defaultdict
import secrets

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vllm_api_server.log")
    ]
)
logger = logging.getLogger(__name__)

# API密钥配置
KEYS_FILE = "api_keys.json"

# 检查密钥文件是否存在
if os.path.exists(KEYS_FILE):
    try:
        with open(KEYS_FILE, 'r') as f:
            keys_data = json.load(f)
            API_KEYS = keys_data.get('keys', [])
        logger.info(f"已加载 {len(API_KEYS)} 个API密钥")
    except Exception as e:
        logger.error(f"从文件加载密钥时出错: {e}")
        # 如果读取失败，生成新密钥
        API_KEYS = [f'sk-{secrets.token_urlsafe(32)}']
        logger.info(f"由于文件读取错误，已生成新的API密钥")
else:
    # 生成新密钥并保存到文件
    API_KEYS = [f'sk-{secrets.token_urlsafe(32)}']
    try:
        with open(KEYS_FILE, 'w') as f:
            json.dump({'keys': API_KEYS}, f)
        logger.info(f"已生成新的API密钥并保存到 {KEYS_FILE}")
    except Exception as e:
        logger.error(f"保存密钥到文件时出错: {e}")

# 输出密钥信息
for idx, key in enumerate(API_KEYS, 1):
    logger.info(f"密钥 #{idx}: {key}")

# API Key认证方式
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# 定义模型配置
class ModelConfig(BaseModel):
    name: str
    url: str
    port: int
    description: str = ""
    owner: str = ""
    created: int = int(datetime.now().timestamp())
    model_path: str = ""  # 添加模型路径字段

# 定义API Key模型
class APIKeyInfo(BaseModel):
    key: str
    description: str
    models: List[str] = []
    rate_limit: int = 10  # 每分钟请求限制

# 定义聊天请求模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

# 定义模型名称到后端URL的映射
MODEL_CONFIGS = {
    "DeepSeek-R1-0528-Qwen3-14B": ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-14B",
        url="http://localhost",
        port=8000,
        description="DeepSeek R1 0528 Qwen3 14B模型",
        owner="DeepSeek",
        model_path="/root/data/model/zjydiary/DeepSeek-R1-0528-Qwen3-14B"
    ),
    "Qwen3-14B": ModelConfig(
        name="Qwen3-14B",
        url="http://localhost",
        port=8001,
        description="Qwen3 14B模型",
        owner="Qwen",
        model_path="/root/data/model/Qwen/Qwen3-14B"
    ),
    "DeepSeek-R1-Distill-Qwen-7B": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-7B",
        url="http://localhost",
        port=8002,
        description="DeepSeek R1 Distill Qwen 7B模型",
        owner="DeepSeek",
        model_path="/root/data/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ),
    "Qwen2.5-7B-Instruct": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        url="http://localhost",
        port=8003,
        description="Qwen2.5 7B Instruct模型",
        owner="Qwen",
        model_path="/root/data/model/Qwen/Qwen2.5-7B-Instruct"
    )
}

# 获取所有可用模型名称
ALL_MODELS = list(MODEL_CONFIGS.keys())

# 创建API Key管理
API_KEY_INFO = {
    key: APIKeyInfo(
        key=key,
        description="通用API密钥",
        models=ALL_MODELS,  # 所有模型都可以使用同一个密钥
        rate_limit=20  # 增加速率限制
    )
    for key in API_KEYS
}

# 模型列表响应
def get_models_list():
    return {
        "object": "list",
        "data": [
            {
                "id": config.name,
                "object": "model",
                "created": config.created,
                "owned_by": config.owner,
                "description": config.description,
                "model_path": config.model_path  # 添加模型路径到响应
            }
            for config in MODEL_CONFIGS.values()
        ]
    }

# 请求计数器
request_counter = defaultdict(int)
last_reset = datetime.now()

def reset_counters():
    global request_counter, last_reset
    now = datetime.now()
    if (now - last_reset).seconds >= 60:
        request_counter = defaultdict(int)
        last_reset = now
        logger.info("已重置速率限制计数器")

async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header)
) -> Optional[str]:
    """获取并验证API Key"""
    # 首先检查查询参数
    if api_key_query and api_key_query in API_KEY_INFO:
        return api_key_query
    
    # 检查请求头
    if api_key_header:
        # 支持 "Bearer <token>" 格式
        if api_key_header.startswith("Bearer "):
            token = api_key_header[7:].strip()
            if token in API_KEY_INFO:
                return token
        # 支持直接使用token
        elif api_key_header in API_KEY_INFO:
            return api_key_header
    
    # 未提供有效API Key
    return None

async def verify_api_key(api_key: str = Depends(get_api_key)):
    """验证API Key是否有效"""
    reset_counters()
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="缺少API密钥。请通过'Authorization: Bearer <key>'请求头或'api_key'查询参数提供",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查速率限制
    key_info = API_KEY_INFO.get(api_key)
    if key_info:
        request_counter[api_key] += 1
        if request_counter[api_key] > key_info.rate_limit:
            raise HTTPException(
                status_code=429,
                detail=f"超出速率限制。每分钟最多{key_info.rate_limit}个请求",
                headers={"Retry-After": "60"},
            )
        return api_key
    
    # API Key无效
    raise HTTPException(
        status_code=401,
        detail="无效的API密钥",
        headers={"WWW-Authenticate": "Bearer"},
    )

# 创建FastAPI应用
app = FastAPI(
    title="VLLM模型API服务",
    description="为多个VLLM模型提供统一的API接口，支持API密钥认证",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "模型",
            "description": "模型信息相关操作"
        },
        {
            "name": "聊天",
            "description": "聊天完成操作"
        },
        {
            "name": "管理",
            "description": "管理操作"
        }
    ]
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def forward_stream(request: Request, backend_url: str, body: dict, headers: dict):
    """实时转发流式响应"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 创建到后端的流式请求
        async with client.stream(
            method=request.method,
            url=backend_url,
            json=body if request.method == "POST" else None,
            params=request.query_params if request.method == "GET" else None,
            headers=headers,
            timeout=300.0,
            follow_redirects=True
        ) as backend_response:
            # 记录流式传输开始
            logger.info(f"开始向客户端转发流")
            
            # 设置内容类型为事件流
            content_type = backend_response.headers.get("content-type", "text/event-stream")
            
            # 实时转发响应
            async for chunk in backend_response.aiter_bytes():
                yield chunk
                # 立即刷新确保及时发送
                await asyncio.sleep(0.001)
            
            # 记录流式传输完成
            logger.info(f"流转发已完成")

@app.get("/v1/models", tags=["模型"])
async def list_models(api_key: str = Depends(verify_api_key)):
    """列出所有可用模型"""
    return get_models_list()

@app.get("/v1/models/{model_id}", tags=["模型"])
async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
    """获取特定模型的信息"""
    if model_id not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"找不到模型 {model_id}")
    
    config = MODEL_CONFIGS[model_id]
    return {
        "id": config.name,
        "object": "model",
        "created": config.created,
        "owned_by": config.owner,
        "description": config.description,
        "model_path": config.model_path  # 添加模型路径到响应
    }

@app.api_route("/v1/{endpoint:path}", methods=["GET", "POST", "OPTIONS", "HEAD"], tags=["模型", "聊天"])
async def proxy_request(
    request: Request, 
    endpoint: str,
    api_key: str = Depends(verify_api_key)
):
    """代理请求到后端模型服务"""
    # 处理OPTIONS预检请求
    if request.method == "OPTIONS":
        return Response(status_code=200)
    
    # 特殊处理 /v1/models 端点
    if endpoint == "models":
        # 如果是查询模型列表，直接返回数据
        if request.method == "GET":
            return get_models_list()
    
    # 读取请求体（仅对POST请求）
    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass
    
    # 获取模型名称
    model_name = body.get("model", "")
    if not model_name:
        # 尝试从查询参数获取模型名称
        model_name = request.query_params.get("model", "")
    
    # 如果仍然没有模型名称，尝试从URL路径推断
    if not model_name and endpoint.startswith("models/"):
        model_name = endpoint.split("/")[1]
    
    # 如果还是没有模型名称，返回错误
    if not model_name:
        raise HTTPException(status_code=400, detail="请求中未提供模型名称")
    
    # 检查API Key是否有权限访问该模型
    key_info = API_KEY_INFO.get(api_key)
    if key_info and model_name not in key_info.models:
        raise HTTPException(
            status_code=403,
            detail=f"API密钥没有访问模型 {model_name} 的权限"
        )

    # 查找后端配置
    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"不支持模型 {model_name}")

    # 构建后端URL
    backend_url = f"{model_config.url}:{model_config.port}"
    
    # 构建转发URL
    target_url = f"{backend_url}/v1/{endpoint}"
    
    # 添加查询参数
    if request.query_params:
        query_string = "&".join([f"{k}={v}" for k, v in request.query_params.items()])
        target_url += f"?{query_string}"

    # 记录请求
    logger.info(f"转发 {request.method} 请求，模型 {model_name} 到 {target_url}")

    # 判断是否是流式请求
    is_stream = body.get("stream", False)
    if "text/event-stream" in request.headers.get("accept", ""):
        is_stream = True

    # 准备请求头
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    try:
        # 对于流式请求，使用实时流式传输
        if is_stream:
            # 创建实时流式响应
            return StreamingResponse(
                content=forward_stream(request, target_url, body, headers),
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                    "X-API-Key": api_key[-4:]  # 返回部分API Key用于调试
                }
            )
        
        # 对于非流式请求
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                proxy_response = await client.request(
                    method=request.method,
                    url=target_url,
                    json=body if request.method == "POST" else None,
                    params=request.query_params if request.method == "GET" else None,
                    headers=headers,
                    timeout=300.0
                )
                
                # 创建普通响应
                return Response(
                    content=proxy_response.content,
                    status_code=proxy_response.status_code,
                    headers={
                        **dict(proxy_response.headers),
                        "X-API-Key": api_key[-4:]  # 返回部分API Key用于调试
                    }
                )
            
    except httpx.ConnectError as e:
        logger.error(f"无法连接到模型 {model_name} 的后端: {str(e)}")
        raise HTTPException(status_code=503, detail="服务不可用")
    except httpx.ReadTimeout as e:
        logger.error(f"连接到模型 {model_name} 的后端超时: {str(e)}")
        raise HTTPException(status_code=504, detail="网关超时")
    except Exception as e:
        logger.error(f"意外错误: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health", tags=["管理"], summary="健康检查")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok", 
        "models": [
            {
                "name": config.name,
                "port": config.port,
                "model_path": config.model_path  # 添加模型路径到健康检查响应
            } 
            for config in MODEL_CONFIGS.values()
        ],
        "api_keys": len(API_KEY_INFO)
    }

@app.get("/admin/keys", tags=["管理"], summary="列出API密钥")
async def list_api_keys(api_key: str = Depends(verify_api_key)):
    """列出API Key信息（简化版）"""
    # 只有特定API Key可以访问此端点
    if api_key != API_KEYS[0]:  # 使用第一个密钥作为管理密钥
        raise HTTPException(
            status_code=403,
            detail="权限不足"
        )
    
    return [
        {
            "key": key_info.key[:4] + "****" + key_info.key[-4:],
            "description": key_info.description,
            "models": key_info.models,
            "rate_limit": key_info.rate_limit
        }
        for key_info in API_KEY_INFO.values()
    ]

def run_server():
    """启动服务器函数"""
    config = uvicorn.Config(
        app="vllm_api_server:app",  # 使用模块:应用格式
        host="0.0.0.0",
        port=1024,
        timeout_keep_alive=300,
        http="httptools",
        workers=32,  # 工作线程数量
        limit_concurrency=100,
        log_level="info",
        reload=False
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_server()