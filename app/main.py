from fastapi import FastAPI, Query, HTTPException
import os
from fastapi_mcp import FastApiMCP
from lib.models import ModelEnum
import lib.endpoints as endpoints
from lib.endpoints import chat_with_braincell, ChatRequest

app = FastAPI()

# Wrap with MCP for Model Context Protocol support
mcp = FastApiMCP(app)

# Mount the MCP server directly to your FastAPI app
mcp.mount()

@app.post("/initialize-server")
async def initialize_server(
    model: ModelEnum,
    threads: int = Query(os.cpu_count() // 2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(2048, gt=0),
    port: int = Query(8081, gt=1023),
    system_prompt: str = Query("You are a helpful assistant.", description="Unique system prompt for this server instance"),
    n_predict: int = Query(4096, gt=0, description="Number of tokens to predict for the server instance"),
    temperature: float = Query(0.8, gt=0.0, le=2.0, description="Temperature for sampling")
):
    return await endpoints.initialize_server_endpoint(
        model=model,
        threads=threads,
        ctx_size=ctx_size,
        port=port,
        system_prompt=system_prompt,
        n_predict=n_predict,
        temperature=temperature
    )

@app.post("/shutdown-server")
async def shutdown_server(port: int = Query(8081, gt=1023)):
    return await endpoints.shutdown_server_endpoint(port=port)

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
    return await endpoints.run_benchmark(model, n_token, threads, n_prompt)

@app.get("/perplexity")
async def perplexity(
    model: ModelEnum,
    prompt: str,
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(4, gt=0),
    ppl_stride: int = Query(0, ge=0)
):
    return await endpoints.run_perplexity(model, prompt, threads, ctx_size, ppl_stride)

@app.get("/model-sizes")
def model_sizes():
    return endpoints.get_model_sizes()

@app.post("/chat")
async def chat(chat: ChatRequest):
    chat_fn = chat_with_braincell(chat)
    return await chat_fn()