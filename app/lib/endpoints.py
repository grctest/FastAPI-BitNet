# --- bitnet Orchestrator (Middleman Proxy) ---
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query, Depends
from .models import ModelEnum, BenchmarkRequest, PerplexityRequest, InferenceRequest
from .utils import run_command, parse_benchmark_data, parse_perplexity_data
import os
import subprocess
import atexit
import time
import httpx

from typing import List
from pydantic import BaseModel, Field
from fastapi import HTTPException
import asyncio

# --- Server Process Management ---
# Each server instance is tracked by a unique (host, port) key
server_processes = {}
server_configs = {}

def _terminate_server_process(key):
    proc = server_processes.get(key)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
    server_processes.pop(key, None)
    server_configs.pop(key, None)

def _terminate_all_servers():
    for key in list(server_processes.keys()):
        _terminate_server_process(key)

atexit.register(_terminate_all_servers)

def _total_threads_in_use():
    return sum(cfg['threads'] for cfg in server_configs.values() if 'threads' in cfg)

def _max_threads():
    return os.cpu_count() or 1

async def initialize_server_endpoint(
    threads: int = Query(1, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(2048, gt=0),
    port: int = Query(8081, gt=8080, le=65535),
    system_prompt: str = Query("You are a helpful assistant.", description="Unique system prompt for this server instance"),
    n_predict: int = Query(256, gt=0, description="Number of tokens to predict for the server instance."),
    temperature: float = Query(0.8, gt=0.0, le=2.0, description="Temperature for sampling")
):
    """
    Initializes a llama-server process in the background if not already running on the given port.
    Will not oversubscribe threads beyond system capacity.
    Allows a unique system prompt per server instance.
    """
    host = "127.0.0.1"
    key = (host, port)
    build_dir = os.getenv("BUILD_DIR", "build")
    server_path = os.path.join(build_dir, "bin", "llama-server")
    if not os.path.exists(server_path):
        raise HTTPException(status_code=500, detail=f"Server binary not found at '{server_path}'")
    # Check if already running
    if key in server_processes and server_processes[key].poll() is None:
        return {"message": f"Server already running on {host}:{port}", "pid": server_processes[key].pid, "config": server_configs[key]}
    # Check thread oversubscription and per-server thread limit
    max_threads = _max_threads()
    if threads > max_threads:
        raise HTTPException(status_code=400, detail=f"Requested threads ({threads}) exceed available CPU threads ({max_threads}).")
    threads_in_use = _total_threads_in_use()
    if threads_in_use + threads > max_threads:
        raise HTTPException(status_code=429, detail=f"Cannot start server: would oversubscribe CPU threads (in use: {threads_in_use}, requested: {threads}, max: {max_threads})")
    command = [
        server_path,
        '-m', "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        '-c', str(ctx_size),
        '-t', str(threads),
        '-n', str(n_predict),
        '-ngl', '0',
        '--temp', str(temperature),
        '--host', host,
        '--port', str(port),
        '-cb',  # Enable continuous batching
    ]
    if system_prompt:
        command += ['-p', system_prompt]
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        time.sleep(2)
        if proc.poll() is not None:
            stderr_output = proc.stderr.read().decode(errors='ignore') if proc.stderr else ''
            proc = None
            raise HTTPException(status_code=500, detail=f"Server failed to start. Stderr: {stderr_output}")
        server_processes[key] = proc
        server_configs[key] = {
            "model": "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            "threads": threads,
            "ctx_size": ctx_size,
            "host": host,
            "port": port,
            "system_prompt": system_prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "pid": proc.pid
        }
        return {"message": f"Server started on {host}:{port}", "pid": proc.pid, "config": server_configs[key]}
    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

async def shutdown_server_endpoint(port: int = Query(8081, gt=1023)):
    host = "127.0.0.1"
    key = (host, port)
    if key in server_processes and server_processes[key].poll() is None:
        pid = server_processes[key].pid
        _terminate_server_process(key)
        return {"message": f"Shutdown initiated for server (PID: {pid}) on {host}:{port}."}
    else:
        _terminate_server_process(key)
        return {"message": f"No running server found on {host}:{port}."}

async def get_server_status(port: int = Query(8081, gt=1023)):
    host = "127.0.0.1"
    key = (host, port)
    proc = server_processes.get(key)
    cfg = server_configs.get(key)
    if proc and proc.poll() is None:
        return {"status": "running", "pid": proc.pid, "config": cfg}
    else:
        _terminate_server_process(key)
        return {"status": "stopped", "config": cfg}

# Benchmark endpoint
async def run_benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    n_prompt: int = Query(32, gt=0)
):
    """Run benchmark on specified model"""
    request = BenchmarkRequest(model=model, n_token=n_token, threads=threads, n_prompt=n_prompt)
    build_dir = os.getenv("BUILD_DIR", "build")
    bench_path = os.path.join(build_dir, "bin", "llama-bench")
    if not os.path.exists(bench_path):
        raise HTTPException(status_code=500, detail="Benchmark binary not found")
    command = [
        bench_path,
        '-m', request.model.value,
        '-n', str(request.n_token),
        '-ngl', '0',
        '-b', '1',
        '-t', str(request.threads),
        '-p', str(request.n_prompt),
        '-r', '5'
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        parsed_data = parse_benchmark_data(result.stdout)
        return parsed_data
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

# Validate prompt length for perplexity

def validate_prompt_length(prompt: str = Query(..., description="Input text for perplexity calculation"), ctx_size: int = Query(10, gt=3)) -> str:
    token_count = len(prompt.split())
    min_tokens = 2 * ctx_size
    if token_count < min_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too short. Needs at least {min_tokens} tokens, got {token_count}"
        )
    return prompt

# Perplexity endpoint
async def run_perplexity(
    model: ModelEnum,
    prompt: str = Depends(validate_prompt_length),
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(10, gt=3),
    ppl_stride: int = Query(0, ge=0)
):
    """Calculate perplexity for given text and model"""
    try:
        request = PerplexityRequest(
            model=model, 
            prompt=prompt, 
            threads=threads, 
            ctx_size=ctx_size, 
            ppl_stride=ppl_stride
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    build_dir = os.getenv("BUILD_DIR", "build")
    ppl_path = os.path.join(build_dir, "bin", "llama-perplexity")
    if not os.path.exists(ppl_path):
        raise HTTPException(status_code=500, detail="Perplexity binary not found")

    command = [
        ppl_path,
        '--model', request.model.value,
        '--prompt', request.prompt,
        '--threads', str(request.threads),
        '--ctx-size', str(request.ctx_size),
        '--perplexity',
        '--ppl-stride', str(request.ppl_stride)
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        parsed_data = parse_perplexity_data(result.stderr)
        return parsed_data
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model sizes endpoint
def get_model_sizes():
    """Endpoint to get the file sizes of supported .gguf models."""
    model_sizes = {}
    models_dir = "models"
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".gguf"):
                    file_path = os.path.join(subdir_path, file)
                    file_size_bytes = os.path.getsize(file_path)
                    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)
                    file_size_gb = round(file_size_bytes / (1024 * 1024 * 1024), 3)
                    model_sizes[file] = {
                        "bytes": file_size_bytes,
                        "MB": file_size_mb,
                        "GB": file_size_gb
                    }
    return model_sizes

class ChatRequest(BaseModel):
    message: str
    port: int = 8081
    threads: int = 1
    ctx_size: int = 2048
    n_predict: int = 256
    temperature: float = 0.8

async def chat_with_bitnet(
    chat: ChatRequest
):
    """
    Middleman endpoint: receives a chat message and forwards it to the specified bitnet (llama server instance) by port.
    Returns the response from the bitnet.
    """
    host = "127.0.0.1"
    key = (host, chat.port)
    proc = server_processes.get(key)
    cfg = server_configs.get(key)
    if not (proc and proc.poll() is None and cfg):
        raise HTTPException(status_code=503, detail=f"bitnet server not running on {host}:{chat.port}. Initialize it first.")
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
                response = await client.post(server_url, json=payload, timeout=180.0)
                response.raise_for_status()
                result_data = response.json()
                content = result_data.get("content", result_data)
                return {"result": content}
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Request to bitnet server timed out.")
            except httpx.ConnectError:
                raise HTTPException(status_code=503, detail=f"Could not connect to bitnet server at {server_url}. Is it running?")
            except httpx.RequestError as e:
                raise HTTPException(status_code=500, detail=f"Error during request to bitnet server: {str(e)}")
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text or str(e)
                raise HTTPException(status_code=e.response.status_code, detail=f"bitnet server returned error: {error_detail}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error during chat: {str(e)}")
    return _chat

class MultiChatRequest(BaseModel):
    requests: List[ChatRequest]

async def multichat_with_bitnet(multichat: MultiChatRequest):
    async def run_chat(chat_req: ChatRequest):
        chat_fn = chat_with_bitnet(chat_req)
        return await chat_fn()
    results = await asyncio.gather(*(run_chat(req) for req in multichat.requests), return_exceptions=True)
    # Format results: if exception, return error message
    formatted = []
    for res in results:
        if isinstance(res, Exception):
            if isinstance(res, HTTPException):
                formatted.append({"error": res.detail, "status_code": res.status_code})
            else:
                formatted.append({"error": str(res)})
        else:
            formatted.append(res)
    return {"results": formatted}