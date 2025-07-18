from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import httpx
import logging
import uvicorn
import json
import asyncio
import os
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embedding_reranker_proxy.log")
    ]
)
logger = logging.getLogger(__name__)

# 从文件读取或生成API密钥
import secrets
import json
import os.path

KEYS_FILE = "er_api_keys.json"

# 检查密钥文件是否存在
if os.path.exists(KEYS_FILE):
    try:
        with open(KEYS_FILE, 'r') as f:
            keys_data = json.load(f)
            API_KEYS = keys_data.get('keys', [])
        logger.info(f"Loaded {len(API_KEYS)} API keys from {KEYS_FILE}")
    except Exception as e:
        logger.error(f"Error loading keys from file: {e}")
        # 如果读取失败，生成新密钥
        API_KEYS = [f'sk-{secrets.token_urlsafe(32)}']
        logger.info(f"Generated new API key due to file read error")
else:
    # 生成新密钥并保存到文件
    API_KEYS = [f'sk-{secrets.token_urlsafe(32)}']
    try:
        with open(KEYS_FILE, 'w') as f:
            json.dump({'keys': API_KEYS}, f)
        logger.info(f"Generated new API key and saved to {KEYS_FILE}")
    except Exception as e:
        logger.error(f"Error saving keys to file: {e}")

# 输出密钥信息
for idx, key in enumerate(API_KEYS, 1):
    logger.info(f"Key #{idx}: {key}")


# API Key认证方式
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# 定义API Key模型
class APIKeyInfo(BaseModel):
    key: str
    description: str
    models: List[str] = []
    rate_limit: int = 10000000000  # 每分钟请求限制

# 创建FastAPI应用
app = FastAPI(
    title="Embedding & Reranker Proxy API",
    description="Proxy server for Qwen3 Embedding and Reranker models with API key authentication",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Models",
            "description": "Operations related to model information"
        },
        {
            "name": "Embeddings",
            "description": "Text embedding operations"
        },
        {
            "name": "Reranking",
            "description": "Text reranking operations"
        },
        {
            "name": "Admin",
            "description": "Administrative operations"
        }
    ]
)

# 添加CORS中间件以处理预检请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义模型名称到后端URL的映射
MODEL_BACKENDS = {
    "Qwen3-Embedding-0.6B": "http://localhost:8000",
    "Qwen3-Reranker-0.6B": "http://localhost:8001"
}

# 获取所有可用模型名称
ALL_MODELS = list(MODEL_BACKENDS.keys())

# 创建API Key管理
API_KEY_INFO = {
    key: APIKeyInfo(
        key=key,
        description="Universal API Key for Embedding & Reranker",
        models=ALL_MODELS,  # 所有模型都可以使用同一个密钥
        rate_limit=300000000  # 大幅增加速率限制
    )
    for key in API_KEYS
}

# 模拟模型列表响应
MODELS_LIST = {
    "object": "list",
    "data": [
        {
            "id": "Qwen3-Embedding-0.6B",
            "object": "model",
            "created": 1700000000,
            "owned_by": "Qwen",
            "type": "embedding"
        },
        {
            "id": "Qwen3-Reranker-0.6B",
            "object": "model",
            "created": 1700000000,
            "owned_by": "Qwen",
            "type": "reranker"
        }
    ]
}

# 请求计数器
from collections import defaultdict
from datetime import datetime
request_counter = defaultdict(int)
last_reset = datetime.now()

def reset_counters():
    global request_counter, last_reset
    now = datetime.now()
    if (now - last_reset).seconds >= 60:
        request_counter = defaultdict(int)
        last_reset = now
        logger.info("Reset rate limit counters")

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
            detail="Missing API Key. Provide via 'Authorization: Bearer <key>' header or 'api_key' query parameter",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查速率限制
    key_info = API_KEY_INFO.get(api_key)
    if key_info:
        request_counter[api_key] += 1
        if request_counter[api_key] > key_info.rate_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {key_info.rate_limit} requests per minute",
                headers={"Retry-After": "60"},
            )
        return api_key
    
    # API Key无效
    raise HTTPException(
        status_code=401,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.api_route("/v1/{endpoint:path}", methods=["GET", "POST", "OPTIONS", "HEAD"], tags=["Models", "Embeddings", "Reranking"])
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
        # 如果是查询模型列表，直接返回模拟数据
        if request.method == "GET":
            return MODELS_LIST
        
        # 如果是查询特定模型
        if request.method == "GET" and request.query_params.get("id"):
            model_id = request.query_params.get("id")
            if model_id in MODEL_BACKENDS:
                return next(m for m in MODELS_LIST["data"] if m["id"] == model_id)
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
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
    
    # 对于embedding和reranking端点，根据端点类型选择默认模型
    if not model_name:
        if endpoint == "embeddings":
            model_name = "Qwen3-Embedding-0.6B"
        elif endpoint == "rerank":
            model_name = "Qwen3-Reranker-0.6B"
        else:
            raise HTTPException(status_code=400, detail="Model name not provided in request")
    
    # 检查API Key是否有权限访问该模型
    key_info = API_KEY_INFO.get(api_key)
    if key_info and model_name not in key_info.models:
        raise HTTPException(
            status_code=403,
            detail=f"API Key does not have access to model {model_name}"
        )

    # 查找后端URL
    backend_url = MODEL_BACKENDS.get(model_name)
    if not backend_url:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    # 构建转发URL
    target_url = f"{backend_url}/v1/{endpoint}"
    
    # 添加查询参数
    if request.query_params:
        query_string = "&".join([f"{k}={v}" for k, v in request.query_params.items()])
        target_url += f"?{query_string}"

    # 记录请求
    logger.info(f"Forwarding {request.method} request for model {model_name} to {target_url}")

    # 准备请求头
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    try:
        # 增加连接池大小和超时时间以提高并发性能
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
            proxy_response = await client.request(
                method=request.method,
                url=target_url,
                json=body if request.method == "POST" else None,
                params=request.query_params if request.method == "GET" else None,
                headers=headers,
                timeout=600.0
            )
            
            # 创建响应
            return Response(
                content=proxy_response.content,
                status_code=proxy_response.status_code,
                headers={
                    **dict(proxy_response.headers),
                    "X-API-Key": api_key[-4:]  # 返回部分API Key用于调试
                }
            )
            
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to backend for model {model_name}: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except httpx.ReadTimeout as e:
        logger.error(f"Timeout connecting to backend for model {model_name}: {str(e)}")
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", tags=["Admin"], summary="Health check")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok", 
        "models": list(MODEL_BACKENDS.keys()),
        "api_keys": len(API_KEY_INFO),
        "service_type": "embedding_reranker_proxy"
    }

@app.get("/admin/keys", tags=["Admin"], summary="List API keys")
async def list_api_keys(api_key: str = Depends(verify_api_key)):
    """列出API Key信息（简化版）"""
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
        app="embedding_reranker_proxy:app",  # 使用模块:应用格式
        host="0.0.0.0",
        port=1024,  # 使用不同的端口避免冲突
        timeout_keep_alive=60000,  # 增加保持连接时间
        http="httptools",
        workers=64,  # 大幅增加工作进程数
        limit_concurrency=10000000000,  # 大幅增加并发限制
        log_level="info",
        reload=False
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_server()