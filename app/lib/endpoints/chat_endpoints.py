from fastapi import HTTPException
import httpx
import asyncio
import logging
from .process_management import get_server_processes, get_server_configs
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    port: int = 8081
    threads: int = 1
    ctx_size: int = 2048
    n_predict: int = 256
    temperature: float = 0.8

async def chat_with_bitnet(chat: ChatRequest):
    host = "127.0.0.1"
    key = (host, chat.port)
    proc_entry = get_server_processes().get(key)
    cfg = get_server_configs().get(key)
    if not (proc_entry and proc_entry["process"].returncode is None and cfg):
        logger.warning(f"Chat request to non-existent or stopped server on port {chat.port}.")
        raise HTTPException(status_code=404, detail=f"Server on port {chat.port} not running or not configured.")
    server_url = f"http://{host}:{chat.port}/completion"
    payload = {
        "prompt": chat.message,
        "threads": chat.threads,
        "ctx_size": chat.ctx_size,
        "n_predict": chat.n_predict,
        "temperature": chat.temperature
    }
    async def _chat():
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Forwarding chat message to BitNet server on port {chat.port}.")
                response = await client.post(server_url, json=payload, timeout=300.0)
                response.raise_for_status()
                return response.json()
            except httpx.ReadTimeout:
                logger.error(f"ReadTimeout when communicating with BitNet server on port {chat.port}.")
                raise HTTPException(status_code=504, detail=f"Request to BitNet server on port {chat.port} timed out.")
            except httpx.ConnectError:
                logger.error(f"ConnectError when communicating with BitNet server on port {chat.port}.")
                raise HTTPException(status_code=503, detail=f"Could not connect to BitNet server on port {chat.port}.")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError from BitNet server on port {chat.port}: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise HTTPException(status_code=e.response.status_code, detail=f"BitNet server error: {e.response.text}")
            except Exception as e:
                logger.error(f"Unexpected error during chat with BitNet server on port {chat.port}: {str(e)}", exc_info=True)
                error_detail = f"An unexpected error occurred while communicating with BitNet server on port {chat.port}: {str(e)}"
                raise HTTPException(status_code=500, detail=error_detail)
    return await _chat()

class MultiChatRequest(BaseModel):
    requests: List[ChatRequest]

async def multichat_with_bitnet(multichat: MultiChatRequest):
    logger.info(f"Multichat request received for {len(multichat.requests)} chats.")
    async def run_chat(chat_req: ChatRequest):
        chat_fn = chat_with_bitnet(chat_req)
        return await chat_fn
    results = await asyncio.gather(*(run_chat(req) for req in multichat.requests), return_exceptions=True)
    formatted = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            if isinstance(res, HTTPException):
                formatted.append({"error": res.detail})
            else:
                formatted.append({"error": str(res)})
        elif isinstance(res, dict) and "content" in res:
            formatted.append(res["content"])
        else:
            formatted.append(res)
    logger.info("Multichat processing completed.")
    return {"results": formatted}
