# --- bitnet Orchestrator (Middleman Proxy) ---
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query, Depends
from .models import ModelEnum, BenchmarkRequest, PerplexityRequest, InferenceRequest
from .utils import run_command, parse_benchmark_data, parse_perplexity_data
import os
import subprocess # Keep for CalledProcessError
import atexit
import httpx
import asyncio # Ensure asyncio is imported

from typing import List
from pydantic import BaseModel, Field
from fastapi import HTTPException
import logging # Import logging

# --- Logging Configuration for this module ---
logger = logging.getLogger(__name__)

# --- Pydantic Models for Batch Initialization ---
class ServerInitConfig(BaseModel):
    threads: int = Field(1, gt=0, description="Number of threads for this server instance.")
    ctx_size: int = Field(2048, gt=0, description="Context size for this server instance.")
    port: int = Field(..., gt=8081, le=65535, description="Port for this server instance. Must be unique and > 8081.")
    system_prompt: str = Field("You are a helpful assistant.", description="Unique system prompt for this server instance.")
    n_predict: int = Field(256, gt=0, description="Number of tokens to predict for this server instance.")
    temperature: float = Field(0.8, gt=0.0, le=2.0, description="Temperature for sampling for this server instance.")

class BatchServerInitRequest(BaseModel):
    servers: List[ServerInitConfig]

class BatchServerPortRequest(BaseModel):
    ports: List[int]

# --- Server Process Management ---
# Each server instance is tracked by a unique (host, port) key
server_processes: dict[tuple[str, int], asyncio.subprocess.Process] = {} # Store asyncio processes
server_configs = {}
FASTAPI_PORT = 8080 # Port used by the FastAPI application itself

# Helper functions for thread management (non-async, as they inspect config or use os)
def _total_threads_in_use():
    return sum(cfg.get('threads', 0) for cfg in server_configs.values())

def _max_threads():
    return os.cpu_count() or 1 # Default to 1 if cpu_count is None

async def _terminate_server_process(key: tuple[str, int]): # Now async
    host, port = key

    if port == FASTAPI_PORT:
        logger.warning(f"Attempt to terminate FastAPI server on port {port} denied.")
        return f"Operation denied: Port {port} is used by the FastAPI application and cannot be terminated via this endpoint."

    proc_to_terminate = server_processes.get(key)

    if not proc_to_terminate:
        server_configs.pop(key, None)
        logger.info(f"No server process found for key {key} (port {port}) during termination attempt.")
        return f"No server process found for key {key} (port {port})."

    # proc.returncode is None if process is running
    if proc_to_terminate.returncode is None:  # Process is currently running
        pid = proc_to_terminate.pid
        logger.info(f"Attempting to terminate server on port {port} (PID: {pid}).")
        try:
            proc_to_terminate.terminate() # Send SIGTERM
            await asyncio.wait_for(proc_to_terminate.wait(), timeout=5.0) # Wait 5 seconds
            logger.info(f"Server on port {port} (PID: {pid}) terminated successfully after SIGTERM.")
            # Pop after successful termination confirmation
            server_processes.pop(key, None)
            server_configs.pop(key, None)
            return f"Server on port {port} (PID: {pid}) terminated successfully."
        except asyncio.TimeoutError:
            logger.warning(f"Server on port {port} (PID: {pid}) did not respond to SIGTERM within timeout. Attempting SIGKILL.")
            try:
                proc_to_terminate.kill() # Send SIGKILL
                await proc_to_terminate.wait() # Wait for kill to complete (usually immediate)
                logger.info(f"Server on port {port} (PID: {pid}) forcefully killed.")
                # Pop after successful kill confirmation
                server_processes.pop(key, None)
                server_configs.pop(key, None)
                return f"Server on port {port} (PID: {pid}) forcefully killed as it did not respond to SIGTERM."
            except Exception as e_kill:
                logger.error(f"Error forcefully killing server on port {port} (PID: {pid}): {str(e_kill)}", exc_info=True)
                # Do not pop if kill failed, process might still be running or in an unknown state
                return f"Error forcefully killing server on port {port} (PID: {pid}): {str(e_kill)}. Process may still be running."
        except Exception as e_term:
            logger.error(f"Error terminating server on port {port} (PID: {pid}) with SIGTERM: {str(e_term)}", exc_info=True)
            # Do not pop if terminate failed unexpectedly
            return f"Error terminating server on port {port} (PID: {pid}): {str(e_term)}. Process may still be running."
    else: # Process was tracked but is not running (already stopped)
        logger.info(f"Server on port {port} was already stopped (return code: {proc_to_terminate.returncode}). Cleaned up tracking.")
        server_processes.pop(key, None)
        server_configs.pop(key, None)
        return f"Server on port {port} was already stopped. Cleaned up tracking."

_atexit_cleanup_completed = False

async def _terminate_all_servers(): # Now async
    global _atexit_cleanup_completed
    if _atexit_cleanup_completed: # Prevent multiple runs if atexit triggers more than once
        return

    logger.info("Attempting to terminate all running server processes asynchronously at exit.")
    keys_to_terminate = list(server_processes.keys())
    
    tasks = [_terminate_server_process(key) for key in keys_to_terminate]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, key in enumerate(keys_to_terminate):
        result = results[i]
        if isinstance(result, Exception):
            logger.error(f"Error during atexit termination for server {key}: {result}", exc_info=result)
        else:
            logger.info(f"Atexit termination for server {key}: {result}")
            
    logger.info("Asynchronous termination of all server processes at exit completed.")
    _atexit_cleanup_completed = True


def _run_async_cleanup_on_exit():
    try:
        # Try to get the current event loop; if it's closed or not set, asyncio.run will create one.
        asyncio.run(_terminate_all_servers())
    except RuntimeError as e:
        if ("cannot schedule new futures after shutdown" in str(e).lower() or \
            "event loop is closed" in str(e).lower()):
            logger.warning(f"Could not run async cleanup at exit because event loop was closed or shutting down: {e}")
        else:
            logger.error(f"Unexpected RuntimeError during atexit async cleanup: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected Exception during atexit async cleanup: {e}", exc_info=True)

atexit.register(_run_async_cleanup_on_exit)

async def initialize_server_endpoint(
    threads: int = Query(1, gt=0), # Removed le=os.cpu_count() as it's checked dynamically
    ctx_size: int = Query(2048, gt=0),
    port: int = Query(8081, gt=8081, le=65535),
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
        logger.error(f"Server binary not found at '{server_path}' during initialize_server_endpoint call.")
        raise HTTPException(status_code=500, detail=f"Server binary not found at '{server_path}'")
    
    existing_proc = server_processes.get(key)
    if existing_proc and existing_proc.returncode is None:
        logger.info(f"Server already running on {host}:{port}. Initialization request ignored.")
        return {"message": f"Server already running on {host}:{port}", "pid": existing_proc.pid, "config": server_configs.get(key)}
    
    # Check thread oversubscription and per-server thread limit
    max_system_threads = _max_threads()
    if threads > max_system_threads:
        logger.warning(f"Requested threads ({threads}) for port {port} exceed available system CPU threads ({max_system_threads}).")
        raise HTTPException(status_code=400, detail=f"Requested threads ({threads}) exceed available CPU threads ({max_system_threads}).")
    
    current_threads_in_use = _total_threads_in_use()
    if current_threads_in_use + threads > max_system_threads:
        logger.warning(
            f"Cannot start server on port {port}: would oversubscribe CPU threads. "
            f"In use: {current_threads_in_use}, requested: {threads}, max: {max_system_threads}"
        )
        raise HTTPException(status_code=429, detail=f"Cannot start server: would oversubscribe CPU threads (in use: {current_threads_in_use}, requested: {threads}, max: {max_system_threads})")

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

    proc = None 
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL, # Use asyncio's DEVNULL
            stderr=asyncio.subprocess.PIPE     # Capture stderr
        )
        
        # Wait a bit for the server to start or fail quickly
        await asyncio.sleep(2.0) # Non-blocking sleep
        
        if proc.returncode is not None: # Process terminated
            stderr_output_bytes = await proc.stderr.read() if proc.stderr else b''
            stderr_output = stderr_output_bytes.decode(errors='ignore')
            logger.error(f"Server on port {port} failed to start. PID: {proc.pid if proc else 'N/A'}, Return Code: {proc.returncode}. Stderr: {stderr_output}")
            # Ensure proc is not added to server_processes if it failed
            raise HTTPException(status_code=500, detail=f"Server failed to start. Stderr: {stderr_output}")
        
        # If process is still running (returncode is None)
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
        logger.info(f"Server started on {host}:{port} with PID {proc.pid}.")
        return {"message": f"Server started on {host}:{port}", "pid": proc.pid, "config": server_configs[key]}
    except Exception as e:
        if proc and proc.returncode is None: # If proc exists and is running
            proc.kill() # Attempt to kill if something went wrong after start
            await proc.wait() # Ensure it's cleaned up
        logger.error(f"Failed to start server on port {port}: {str(e)}", exc_info=True)
        # Remove from tracking if it was partially added and then an error occurred
        server_processes.pop(key, None)
        server_configs.pop(key, None)
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

async def shutdown_server_endpoint(port: int = Query(8081, gt=1023)):
    host = "127.0.0.1"
    key = (host, port)
    
    message = await _terminate_server_process(key) # Now awaited

    if "terminated successfully" in message or "forcefully killed" in message or "already stopped" in message:
        return {"message": message}
    elif "Operation denied" in message:
        raise HTTPException(status_code=403, detail=message) # Forbidden
    elif "No server process found" in message:
        raise HTTPException(status_code=404, detail=message) # Not Found
    else: # Other errors during termination, e.g., "Error terminating server", "Error forcefully killing server"
        raise HTTPException(status_code=500, detail=message) # Internal Server Error

async def get_server_status(port: int = Query(8081, gt=1023)):
    host = "127.0.0.1"
    key = (host, port)
    proc = server_processes.get(key)
    cfg = server_configs.get(key)

    # proc.returncode is None if running, otherwise it has an exit code
    if proc and proc.returncode is None:
        return {"status": "running", "pid": proc.pid, "config": cfg}
    else:
        # If process is gone or never existed, _terminate_server_process will clean up tracking.
        await _terminate_server_process(key) # Await the cleanup
        return {"status": "stopped", "config": cfg if cfg else f"No configuration found for port {port}."}

# --- New Batch Server Initialization Endpoint ---
async def initialize_batch_servers_endpoint(request: BatchServerInitRequest):
    results = []
    host = "127.0.0.1"
    build_dir = os.getenv("BUILD_DIR", "build")
    server_path = os.path.join(build_dir, "bin", "llama-server")
    logger.info(f"Batch server initialization request received for {len(request.servers)} servers.")

    if not os.path.exists(server_path):
        logger.error(f"Server binary not found at '{server_path}' during batch initialization.")
        raise HTTPException(status_code=500, detail=f"Server binary not found at '{server_path}'")

    ports_in_batch = [s.port for s in request.servers]
    if len(ports_in_batch) != len(set(ports_in_batch)):
        logger.warning("Duplicate ports detected in batch server initialization request.")
        raise HTTPException(status_code=400, detail="Duplicate ports detected in the batch request. All ports must be unique.")

    max_system_threads = _max_threads()
    current_threads_already_in_use = _total_threads_in_use()
    requested_batch_threads = sum(s.threads for s in request.servers)

    if current_threads_already_in_use + requested_batch_threads > max_system_threads:
        available_for_batch = max_system_threads - current_threads_already_in_use
        logger.warning(
            f"Batch request exceeds CPU thread capacity. "
            f"Currently in use: {current_threads_already_in_use}, "
            f"Batch requested: {requested_batch_threads}, "
            f"System maximum: {max_system_threads}. "
            f"Available for batch: {available_for_batch}."
        )
        raise HTTPException(
            status_code=429, 
            detail=(
                f"Batch request exceeds CPU thread capacity. "
                f"Currently in use: {current_threads_already_in_use}, "
                f"Batch requested: {requested_batch_threads}, "
                f"System maximum: {max_system_threads}. "
                f"Available for batch: {available_for_batch}."
            )
        )
    
    for server_config_item in request.servers: # Renamed config to avoid conflict
        # Individual server thread check (should be caught by batch check, but good for safety)
        if server_config_item.threads > max_system_threads:
            logger.warning(f"Server config for port {server_config_item.port} requests {server_config_item.threads} threads, exceeding system max {max_system_threads}. Skipping.")
            results.append({
                "port": server_config_item.port,
                "requested_config": server_config_item.dict(),
                "status": "error",
                "message": f"Requested threads ({server_config_item.threads}) exceed system maximum ({max_system_threads})."
            })
            continue

        key = (host, server_config_item.port)
        operation_status = {
            "port": server_config_item.port,
            "requested_config": server_config_item.dict()
        }
        
        existing_proc = server_processes.get(key)
        if existing_proc and existing_proc.returncode is None:
            logger.info(f"Server already running on {host}:{server_config_item.port} (PID: {existing_proc.pid}). Skipping in batch initialization.")
            operation_status.update({
                "status": "error",
                "message": f"Server already running on {host}:{server_config_item.port}",
                "pid": existing_proc.pid,
                "existing_config": server_configs.get(key)
            })
            results.append(operation_status)
            continue

        command = [
            server_path,
            '-m', "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            '-c', str(server_config_item.ctx_size),
            '-t', str(server_config_item.threads),
            '-n', str(server_config_item.n_predict),
            '-ngl', '0',
            '--temp', str(server_config_item.temperature),
            '--host', host,
            '--port', str(server_config_item.port),
            '-cb',
        ]
        if server_config_item.system_prompt:
            command += ['-p', server_config_item.system_prompt]

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.sleep(2.0) # Wait for server to start or fail

            if proc.returncode is not None:  
                stderr_bytes = await proc.stderr.read() if proc.stderr else b''
                stderr_output = stderr_bytes.decode(errors='ignore')
                logger.error(f"Server on port {server_config_item.port} failed to start during batch init. PID: {proc.pid if proc else 'N/A'}, RC: {proc.returncode}. Stderr: {stderr_output}")
                operation_status.update({
                    "status": "error",
                    "message": f"Server on port {server_config_item.port} failed to start. Stderr: {stderr_output}"
                })
            else: 
                server_processes[key] = proc
                server_configs[key] = {
                    "model": "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
                    "threads": server_config_item.threads,
                    "ctx_size": server_config_item.ctx_size,
                    "host": host,
                    "port": server_config_item.port,
                    "system_prompt": server_config_item.system_prompt,
                    "n_predict": server_config_item.n_predict,
                    "temperature": server_config_item.temperature,
                    "pid": proc.pid
                }
                logger.info(f"Server started on {host}:{server_config_item.port} (PID: {proc.pid}) as part of batch initialization.")
                operation_status.update({
                    "status": "success",
                    "message": f"Server started on {host}:{server_config_item.port}",
                    "pid": proc.pid,
                    "config": server_configs[key]
                })
        except Exception as e:
            if proc and proc.returncode is None: 
                proc.kill()
                await proc.wait()
            logger.error(f"Exception during batch server start for port {server_config_item.port}: {str(e)}", exc_info=True)
            server_processes.pop(key, None) 
            server_configs.pop(key, None)
            operation_status.update({
                "status": "error",
                "message": f"An exception occurred while trying to start server on port {server_config_item.port}: {str(e)}"
            })
        
        results.append(operation_status)

    logger.info(f"Batch server initialization attempt completed. Results: {results}")
    return results

# --- New Batch Server Shutdown Endpoint ---
async def shutdown_batch_servers_endpoint(request: BatchServerPortRequest):
    results = []
    host = "127.0.0.1"
    logger.info(f"Batch server shutdown request received for ports: {request.ports}")
    for port_num in request.ports:
        key = (host, port_num)
        message = await _terminate_server_process(key) # Now awaited
        
        status_code_str = "unknown" # Renamed from status_code to avoid conflict
        if "terminated successfully" in message or "forcefully killed" in message:
            status_code_str = "success_terminated"
        elif "already stopped" in message:
            status_code_str = "success_already_stopped"
        elif "Operation denied" in message:
            status_code_str = "denied"
        elif "No server process found" in message:
            status_code_str = "not_found"
        elif "Error" in message: # Catches "Error terminating" and "Error forcefully killing"
            status_code_str = "error_termination_failed"
            
        results.append({"port": port_num, "status": status_code_str, "message": message})
    logger.info(f"Batch server shutdown attempt completed. Results: {results}")
    return results

# --- New Batch Server Status Endpoint ---
async def get_batch_server_status_endpoint(request: BatchServerPortRequest):
    results = []
    host = "127.0.0.1"
    logger.info(f"Batch server status request received for ports: {request.ports}")
    for port_num in request.ports:
        key = (host, port_num)
        proc = server_processes.get(key)
        cfg = server_configs.get(key)
        if proc and proc.returncode is None: # Check if running
            results.append({"port": port_num, "status": "running", "pid": proc.pid, "config": cfg})
        else:
            await _terminate_server_process(key) # Await cleanup
            results.append({"port": port_num, "status": "stopped", "config": cfg if cfg else f"No configuration found for port {port_num}."})
    logger.info(f"Batch server status attempt completed. Results: {results}")
    return results

# Benchmark endpoint
async def run_benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0), # Removed le=os.cpu_count()
    n_prompt: int = Query(32, gt=0)
):
    """Run benchmark on specified model"""
    request = BenchmarkRequest(model=model, n_token=n_token, threads=threads, n_prompt=n_prompt)
    build_dir = os.getenv("BUILD_DIR", "build")
    bench_path = os.path.join(build_dir, "bin", "llama-bench")
    if not os.path.exists(bench_path):
        logger.error(f"Benchmark binary not found at '{bench_path}'.")
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
        logger.info(f"Running benchmark with command: {' '.join(command)}")
        # Replace subprocess.run with asyncio.create_subprocess_exec and communicate
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await process.communicate() # Wait for completion

        if process.returncode != 0:
            logger.error(f"Benchmark failed. RC: {process.returncode}. Stderr: {stderr_bytes.decode(errors='ignore')}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd=command, output=stdout_bytes, stderr=stderr_bytes
            )
        
        parsed_data = parse_benchmark_data(stdout_bytes.decode(errors='ignore'))
        logger.info("Benchmark completed successfully.")
        return parsed_data
    except subprocess.CalledProcessError as e: # Catch the specific error
        # Log details from the CalledProcessError object
        logger.error(f"Benchmark failed: {str(e)}. Command: {e.cmd}. RC: {e.returncode}. Stdout: {e.stdout.decode(errors='ignore') if e.stdout else ''}. Stderr: {e.stderr.decode(errors='ignore') if e.stderr else ''}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during benchmark: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during benchmark: {str(e)}")

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
    threads: int = Query(2, gt=0), # Removed le=os.cpu_count()
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
        logger.error(f"Perplexity binary not found at '{ppl_path}'.")
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
        logger.info(f"Running perplexity calculation with command: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE, # Perplexity might output to stdout or stderr
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Perplexity calculation failed. RC: {process.returncode}. Stderr: {stderr_bytes.decode(errors='ignore')}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd=command, output=stdout_bytes, stderr=stderr_bytes
            )
        
        # Original code parsed from stderr, stick to that unless known otherwise
        parsed_data = parse_perplexity_data(stderr_bytes.decode(errors='ignore'))
        logger.info("Perplexity calculation completed successfully.")
        return parsed_data
    except subprocess.CalledProcessError as e:
        logger.error(f"Perplexity calculation failed: {str(e)}. Command: {e.cmd}. RC: {e.returncode}. Stderr: {e.stderr.decode(errors='ignore') if e.stderr else ''}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Perplexity calculation failed: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during perplexity calculation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during perplexity calculation: {str(e)}")

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
    # Adapted for asyncio.subprocess.Process: use returncode instead of poll()
    if not (proc and proc.returncode is None and cfg):
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
    # Use httpx for async requests
    async def _chat():
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Forwarding chat message to BitNet server on port {chat.port}.")
                response = await client.post(server_url, json=payload, timeout=60.0) # Increased timeout
                response.raise_for_status()  # Raise an exception for bad status codes
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
    # By default, only return the 'content' field from each response, unless extra_info is requested
    # Optionally, allow extra info via a query param or request field in the future
    formatted = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            # Only return error message
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