from fastapi import FastAPI, Query, HTTPException
import os
from fastapi_mcp import FastApiMCP
from lib.models import ModelEnum
import lib.endpoints as endpoints
from lib.endpoints import chat_with_bitnet, ChatRequest, multichat_with_bitnet, MultiChatRequest
import traceback

app = FastAPI()

@app.post("/initialize-server")
async def initialize_server(
    threads: int = Query(os.cpu_count() // 2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(2048, gt=0),
    port: int = Query(8081, gt=1023),
    system_prompt: str = Query("You are a helpful assistant.", description="Unique system prompt for this server instance"),
    n_predict: int = Query(4096, gt=0, description="Number of tokens to predict for the server instance"),
    temperature: float = Query(0.8, gt=0.0, le=2.0, description="Temperature for sampling")
):
    try:
        return await endpoints.initialize_server_endpoint(
            threads=threads,
            ctx_size=ctx_size,
            port=port,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def _max_threads():
    return os.cpu_count() or 1

# --- Server Initialization and Shutdown Endpoints ---
def validate_thread_allocation(requests):
    max_threads = _max_threads()
    total_requested = sum(req["threads"] for req in requests)
    for req in requests:
        if req["threads"] > max_threads:
            raise HTTPException(
                status_code=400,
                detail=f"Requested {req['threads']} threads for a server, but only {max_threads} are available."
            )
    if total_requested > max_threads:
        raise HTTPException(
            status_code=400,
            detail=f"Total requested threads ({total_requested}) exceed available threads ({max_threads})."
        )

@app.post("/shutdown-server")
async def shutdown_server(port: int = Query(8081, gt=1023)):
    try:
        return await endpoints.shutdown_server_endpoint(port=port)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/server-status")
async def server_status_endpoint(port: int = Query(8081, gt=1023)): # Renamed for clarity
    return await endpoints.get_server_status(port=port) # Call the function from endpoints.py

@app.get("/benchmark")
async def benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    n_prompt: int = Query(32, gt=0)
):
    try:
        return await endpoints.run_benchmark(model, n_token, threads, n_prompt)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/perplexity")
async def perplexity(
    model: ModelEnum,
    prompt: str,
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(4, gt=0),
    ppl_stride: int = Query(0, ge=0)
):
    try:
        return await endpoints.run_perplexity(model, prompt, threads, ctx_size, ppl_stride)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-sizes")
def model_sizes():
    return endpoints.get_model_sizes()

@app.post("/chat")
async def chat(chat: ChatRequest):
    try:
        return await chat_with_bitnet(chat)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Parallel multi-chat endpoint
@app.post("/multichat")
async def multichat(multichat: MultiChatRequest):
    try:
        return await multichat_with_bitnet(multichat)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Wrap with MCP for Model Context Protocol support
mcp = FastApiMCP(app)

# Mount the MCP server directly to your FastAPI app
mcp.mount()