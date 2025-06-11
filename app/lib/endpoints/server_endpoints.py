from fastapi import HTTPException, Query
import os
import psutil
import asyncio
import logging
from .process_management import get_server_processes, get_server_configs, _max_server_instances_by_ram, _terminate_server_process
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
server_processes = get_server_processes()
server_configs = get_server_configs()
FASTAPI_PORT = 8080

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

async def initialize_server_endpoint(
    threads: int = Query(1, gt=0),
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
    
    proc_entry = server_processes.get(key)
    if proc_entry and proc_entry["process"].returncode is None:
        logger.info(f"Server already running on {host}:{port}. Initialization request ignored.")
        return {"message": f"Server already running on {host}:{port}", "pid": proc_entry["pid"], "config": server_configs.get(key)}
    
    max_instances = _max_server_instances_by_ram(1)
    running_instances = len([proc for proc in server_processes.values() if proc["process"].returncode is None])
    if running_instances >= max_instances:
        logger.warning(f"Cannot start server on port {port}: would exceed RAM-based server instance limit. Running: {running_instances}, Max allowed: {max_instances}")
        raise HTTPException(status_code=429, detail=f"Cannot start server: would exceed RAM-based server instance limit (running: {running_instances}, max allowed: {max_instances})")
    
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
        '-cb',
    ]

    if system_prompt:
        command += ['-p', system_prompt]

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )

        await asyncio.sleep(2.0)

        if proc.returncode is not None:
            stderr_output_bytes = await proc.stderr.read() if proc.stderr else b''
            stderr_output = stderr_output_bytes.decode(errors='ignore')
            logger.error(f"Server on port {port} failed to start. PID: {proc.pid if proc else 'N/A'}, Return Code: {proc.returncode}. Stderr: {stderr_output}")
            raise HTTPException(status_code=500, detail=f"Server failed to start. Stderr: {stderr_output}")
        
        server_processes[key] = {"process": proc, "pid": proc.pid}
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
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        logger.error(f"Failed to start server on port {port}: {str(e)}", exc_info=True)
        server_processes.pop(key, None)
        server_configs.pop(key, None)
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

# ...existing code for shutdown_server_endpoint, get_server_status, batch endpoints...


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
    proc_entry = server_processes.get(key)
    cfg = server_configs.get(key)
    # proc.returncode is None if running, otherwise it has an exit code
    if proc_entry and proc_entry["process"].returncode is None:
        # Get RAM usage in MB
        try:
            process = psutil.Process(proc_entry["pid"])
            ram_mb = round(process.memory_info().rss / (1024 ** 2), 2)
        except Exception:
            ram_mb = None
        return {"status": "running", "pid": proc_entry["pid"], "ram_mb": ram_mb, "config": cfg}
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

    # RAM-based limit: check available RAM before starting a batch of new servers
    max_instances = _max_server_instances_by_ram(1)  # 1GB per server
    running_instances = len([proc for proc in server_processes.values() if proc.returncode is None])
    requested_batch_instances = len(request.servers)
    if running_instances + requested_batch_instances > max_instances:
        available_for_batch = max_instances - running_instances
        logger.warning(
            f"Batch request exceeds RAM-based server instance limit. "
            f"Currently running: {running_instances}, "
            f"Batch requested: {requested_batch_instances}, "
            f"RAM-based max: {max_instances}. "
            f"Available for batch: {available_for_batch}."
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Batch request exceeds RAM-based server instance limit. "
                f"Currently running: {running_instances}, "
                f"Batch requested: {requested_batch_instances}, "
                f"RAM-based max: {max_instances}. "
                f"Available for batch: {available_for_batch}."
            )
        )
    
    for server_config_item in request.servers:
        # Individual server thread check
        if server_config_item.threads > os.cpu_count():
            logger.warning(f"Server config for port {server_config_item.port} requests {server_config_item.threads} threads, exceeding system max. Skipping.")
            results.append({
                "port": server_config_item.port,
                "requested_config": server_config_item.dict(),
                "status": "error",
                "message": f"Requested threads ({server_config_item.threads}) exceed system maximums."
            })
            continue

        key = (host, server_config_item.port)
        operation_status = {
            "port": server_config_item.port,
            "requested_config": server_config_item.dict()
        }
        
        proc_entry = server_processes.get(key)
        if proc_entry and proc_entry["process"].returncode is None:
            logger.info(f"Server already running on {host}:{server_config_item.port} (PID: {proc_entry['pid']}). Skipping in batch initialization.")
            operation_status.update({
                "status": "error",
                "message": f"Server already running on {host}:{server_config_item.port}",
                "pid": proc_entry["pid"],
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
                server_processes[key] = {"process": proc, "pid": proc.pid}
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
                # Get RAM usage in MB for the new process
                try:
                    process = psutil.Process(proc.pid)
                    ram_mb = round(process.memory_info().rss / (1024 ** 2), 2)
                except Exception:
                    ram_mb = None
                logger.info(f"Server started on {host}:{server_config_item.port} (PID: {proc.pid}) as part of batch initialization.")
                operation_status.update({
                    "status": "success",
                    "message": f"Server started on {host}:{server_config_item.port}",
                    "pid": proc.pid,
                    "ram_mb": ram_mb,
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
        proc_entry = server_processes.get(key)
        cfg = server_configs.get(key)
        # Fix: proc_entry is a dict with 'process' and 'pid', not a process object directly
        if proc_entry and proc_entry["process"].returncode is None:
            results.append({"port": port_num, "status": "running", "pid": proc_entry["pid"], "config": cfg})
        else:
            await _terminate_server_process(key) # Await cleanup
            results.append({"port": port_num, "status": "stopped", "config": cfg if cfg else f"No configuration found for port {port_num}."})
    logger.info(f"Batch server status attempt completed. Results: {results}")
    return results
