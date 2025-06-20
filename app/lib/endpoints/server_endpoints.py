from fastapi import HTTPException
import os
import psutil
import asyncio
import logging
from .process_management import get_server_processes, get_server_configs, _max_server_instances_by_ram, _terminate_server_process, get_cli_chat_sessions
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Pydantic Models for API contracts ---

class ServerConfig(BaseModel): # Renamed from ServerInitConfig for clarity, as it's used in responses too
    model: str
    threads: int
    ctx_size: int
    host: str
    port: int
    system_prompt: str
    n_predict: int
    temperature: float
    pid: Optional[int] = None

class ServerInitResponse(BaseModel):
    message: str
    pid: Optional[int] = None
    config: Optional[ServerConfig] = None # Changed from dict to ServerConfig

class ServerStatusResponse(BaseModel):
    port: int # Added port to the response model itself
    status: str
    pid: Optional[int] = None
    ram_mb: Optional[float] = None
    config: Optional[ServerConfig] = None # Changed from dict to ServerConfig
    detail: Optional[str] = None

class ServerTerminationResponse(BaseModel):
    port: int # Added port
    message: str
    status: Optional[str] = None # To give more context like "terminated", "not_found"

# Input models (already mostly defined in previous refactoring step)
class ServerInitConfig(BaseModel): # This is the input model for a single server's desired config
    threads: int = Field(default_factory=lambda: os.cpu_count() or 1, gt=0, description="Number of threads for this server instance.")
    ctx_size: int = Field(default=2048, gt=0, description="Context size for this server instance.")
    port: int = Field(..., gt=1023, le=65535, description="Port for this server instance. Must be unique and > 1023.")
    system_prompt: str = Field(default="You are a helpful assistant.", description="Unique system prompt for this server instance.")
    n_predict: int = Field(default=256, gt=0, description="Number of tokens to predict for this server instance.")
    temperature: float = Field(default=0.8, gt=0.0, le=2.0, description="Temperature for sampling for this server instance.")
    # model_path: str = Field(default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf", description="Path to the model GGUF file.") # Model is hardcoded for now

class SingleServerInitRequest(ServerInitConfig): # For single server initialization endpoint
    pass

class BatchServerInitRequest(BaseModel):
    servers: List[ServerInitConfig]

class BatchServerPortRequest(BaseModel):
    ports: List[int]

# --- Handler Functions ---

async def handle_initialize_server(config: SingleServerInitRequest) -> ServerInitResponse: # Added return type hint
    host = "127.0.0.1"
    key = (host, config.port)
    build_dir = os.getenv("BUILD_DIR", "build")
    server_path = os.path.join(build_dir, "bin", "llama-server")
    model_gguf_path = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf" # Hardcoded

    if not os.path.exists(server_path):
        logger.error(f"Server binary not found at '{server_path}' during initialize_server call.")
        raise HTTPException(status_code=500, detail=f"Server binary not found at '{server_path}'")
    if not os.path.exists(model_gguf_path):
        logger.error(f"Model file not found at '{model_gguf_path}' during initialize_server call.")
        raise HTTPException(status_code=500, detail=f"Model file not found at '{model_gguf_path}'")

    current_server_processes = get_server_processes()
    current_server_configs = get_server_configs()

    proc_entry = current_server_processes.get(key)
    if proc_entry and proc_entry["process"].returncode is None:
        logger.info(f"Server already running on {host}:{config.port}. Initialization request ignored.")
        # Ensure the config being returned is of ServerConfig type
        existing_config_data = current_server_configs.get(key)
        return ServerInitResponse(
            message=f"Server already running on {host}:{config.port}",
            pid=proc_entry["pid"],
            config=ServerConfig(**existing_config_data) if existing_config_data else None
        )

    max_instances = _max_server_instances_by_ram(1)
    running_instances = len([proc for proc in current_server_processes.values() if proc["process"].returncode is None])
    if running_instances >= max_instances:
        logger.warning(f"Cannot start server on port {config.port}: would exceed RAM-based server instance limit. Running: {running_instances}, Max allowed: {max_instances}")
        raise HTTPException(status_code=429, detail=f"Cannot start server: would exceed RAM-based server instance limit (running: {running_instances}, max allowed: {max_instances})")

    command = [
        server_path,
        '-m', model_gguf_path,
        '-c', str(config.ctx_size),
        '-t', str(config.threads),
        '-n', str(config.n_predict),
        '-ngl', '0',
        '--temp', str(config.temperature),
        '--host', host,
        '--port', str(config.port),
        '-cb',
    ]
    if config.system_prompt:
        command += ['-p', config.system_prompt]

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
            logger.error(f"Server on port {config.port} failed to start. PID: {proc.pid if proc else 'N/A'}, Return Code: {proc.returncode}. Stderr: {stderr_output}")
            raise HTTPException(status_code=500, detail=f"Server failed to start. Stderr: {stderr_output}")

        server_config_data = {
            "model": model_gguf_path,
            "threads": config.threads,
            "ctx_size": config.ctx_size,
            "host": host,
            "port": config.port,
            "system_prompt": config.system_prompt,
            "n_predict": config.n_predict,
            "temperature": config.temperature,
            "pid": proc.pid
        }
        current_server_processes[key] = {"process": proc, "pid": proc.pid}
        current_server_configs[key] = server_config_data
        logger.info(f"Server started on {host}:{config.port} with PID {proc.pid}.")
        return ServerInitResponse(
            message=f"Server started on {host}:{config.port}",
            pid=proc.pid,
            config=ServerConfig(**server_config_data)
        )
    except Exception as e:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        logger.error(f"Failed to start server on port {config.port}: {str(e)}", exc_info=True)
        current_server_processes.pop(key, None)
        current_server_configs.pop(key, None)
        # Ensure a valid HTTPException is raised for FastAPI to handle
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

async def handle_shutdown_server(port: int) -> ServerTerminationResponse: # Added return type hint
    host = "127.0.0.1"
    key = (host, port)
    message = await _terminate_server_process(key)
    status = "unknown"
    if "terminated successfully" in message or "forcefully killed" in message:
        status = "terminated"
    elif "already stopped" in message:
        status = "already_stopped"
    elif "No server process found" in message:
        status = "not_found"
        raise HTTPException(status_code=404, detail=message)
    elif "Operation denied" in message: # Should not happen
        status = "denied"
        raise HTTPException(status_code=403, detail=message)
    else: # Other errors
        status = "error"
        raise HTTPException(status_code=500, detail=message)
    
    return ServerTerminationResponse(port=port, message=message, status=status)

async def handle_get_server_status(port: int) -> ServerStatusResponse: # Added return type hint
    host = "127.0.0.1"
    key = (host, port)
    current_server_processes = get_server_processes()
    current_server_configs = get_server_configs()
    proc_entry = current_server_processes.get(key)
    cfg_data = current_server_configs.get(key)
    cfg_object = ServerConfig(**cfg_data) if cfg_data else None

    if proc_entry and proc_entry["process"].returncode is None:
        try:
            process = psutil.Process(proc_entry["pid"])
            ram_mb = round(process.memory_info().rss / (1024 ** 2), 2)
            return ServerStatusResponse(port=port, status="running", pid=proc_entry["pid"], ram_mb=ram_mb, config=cfg_object)
        except Exception:
            ram_mb = None
            await _terminate_server_process(key)
            return ServerStatusResponse(port=port, status="stopped", config=cfg_object, detail="Process info unavailable, marked as stopped.")
    else:
        await _terminate_server_process(key)
        if not cfg_object and not proc_entry : # If no config and no process entry, it was never there or fully cleaned.
             raise HTTPException(status_code=404, detail=f"Server on port {port} not found.")
        return ServerStatusResponse(port=port, status="stopped", config=cfg_object, detail=f"Server on port {port} is not running.")

async def handle_initialize_batch_servers(request: BatchServerInitRequest) -> List[ServerInitResponse]: # Added return type hint
    results = []
    host = "127.0.0.1"
    build_dir = os.getenv("BUILD_DIR", "build")
    server_path = os.path.join(build_dir, "bin", "llama-server")
    model_gguf_path = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf" # Hardcoded
    logger.info(f"Batch server initialization request received for {len(request.servers)} servers.")

    if not os.path.exists(server_path):
        logger.error(f"Server binary not found at '{server_path}' during batch initialization.")
        raise HTTPException(status_code=500, detail=f"Server binary not found at '{server_path}'")
    if not os.path.exists(model_gguf_path):
        logger.error(f"Model file not found at '{model_gguf_path}' during batch server initialization.")
        raise HTTPException(status_code=500, detail=f"Model file not found at '{model_gguf_path}'")

    ports_in_batch = [s.port for s in request.servers]
    if len(ports_in_batch) != len(set(ports_in_batch)):
        logger.warning("Duplicate ports detected in batch server initialization request.")
        raise HTTPException(status_code=400, detail="Duplicate ports detected in the batch request. All ports must be unique.")

    current_server_processes = get_server_processes()
    current_server_configs = get_server_configs()
    max_instances = _max_server_instances_by_ram(1)
    running_instances = len([p for p in current_server_processes.values() if p["process"].returncode is None])
    
    new_servers_to_start_count = 0
    for server_config_item in request.servers:
        key_check = (host, server_config_item.port)
        if not (current_server_processes.get(key_check) and current_server_processes[key_check]["process"].returncode is None):
            new_servers_to_start_count += 1

    if running_instances + new_servers_to_start_count > max_instances:
        available_for_batch = max_instances - running_instances
        detail_msg = (
            f"Batch request exceeds RAM-based server instance limit. "
            f"Currently running: {running_instances}, New requested: {new_servers_to_start_count}, Max: {max_instances}. "
            f"Can start {available_for_batch} more."
        )
        logger.warning(detail_msg)
        raise HTTPException(status_code=429, detail=detail_msg)

    for server_config_item in request.servers:
        key = (host, server_config_item.port)
        # Default to error, update on success/skip
        response_item = ServerInitResponse(message=f"Initialization failed for port {server_config_item.port}")

        if server_config_item.threads > (os.cpu_count() or 1):
            msg = f"Requested threads ({server_config_item.threads}) exceed system maximum ({os.cpu_count()})."
            logger.warning(f"Port {server_config_item.port}: {msg}")
            response_item.message = msg
            results.append(response_item)
            continue
        
        proc_entry = current_server_processes.get(key)
        if proc_entry and proc_entry["process"].returncode is None:
            msg = f"Server already running on {host}:{server_config_item.port}"
            logger.info(msg)
            existing_config_data = current_server_configs.get(key)
            response_item.message = msg
            response_item.pid = proc_entry["pid"]
            response_item.config = ServerConfig(**existing_config_data) if existing_config_data else None
            results.append(response_item)
            continue

        command = [
            server_path, '-m', model_gguf_path, '-c', str(server_config_item.ctx_size),
            '-t', str(server_config_item.threads), '-n', str(server_config_item.n_predict),
            '-ngl', '0', '--temp', str(server_config_item.temperature),
            '--host', host, '--port', str(server_config_item.port), '-cb',
        ]
        if server_config_item.system_prompt:
            command += ['-p', server_config_item.system_prompt]

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            await asyncio.sleep(2.0)

            if proc.returncode is not None:
                stderr_bytes = await proc.stderr.read() if proc.stderr else b''
                stderr_output = stderr_bytes.decode(errors='ignore')
                msg = f"Server on port {server_config_item.port} failed to start. Stderr: {stderr_output}"
                logger.error(msg)
                response_item.message = msg
            else:
                server_config_data = {
                    "model": model_gguf_path, "threads": server_config_item.threads,
                    "ctx_size": server_config_item.ctx_size, "host": host, "port": server_config_item.port,
                    "system_prompt": server_config_item.system_prompt, "n_predict": server_config_item.n_predict,
                    "temperature": server_config_item.temperature, "pid": proc.pid
                }
                current_server_processes[key] = {"process": proc, "pid": proc.pid}
                current_server_configs[key] = server_config_data
                msg = f"Server started on {host}:{server_config_item.port}"
                logger.info(f"{msg} with PID {proc.pid}.")
                response_item.message = msg
                response_item.pid = proc.pid
                response_item.config = ServerConfig(**server_config_data)
        except Exception as e:
            if proc and proc.returncode is None:
                proc.kill(); await proc.wait()
            logger.error(f"Exception during batch server start for port {server_config_item.port}: {e}", exc_info=True)
            current_server_processes.pop(key, None); current_server_configs.pop(key, None)
            response_item.message = f"Exception for port {server_config_item.port}: {e}"
        results.append(response_item)
    return results

async def handle_shutdown_batch_servers(request: BatchServerPortRequest) -> List[ServerTerminationResponse]: # Added return type hint
    results = []
    host = "127.0.0.1"
    logger.info(f"Batch server shutdown request received for ports: {request.ports}")
    for port_num in request.ports:
        key = (host, port_num)
        message = await _terminate_server_process(key)
        status_str = "unknown"
        if "terminated successfully" in message or "forcefully killed" in message: status_str = "terminated"
        elif "already stopped" in message: status_str = "already_stopped"
        elif "Operation denied" in message: status_str = "denied"
        elif "No server process found" in message: status_str = "not_found"
        elif "Error" in message: status_str = "error_termination_failed"
        results.append(ServerTerminationResponse(port=port_num, message=message, status=status_str))
    return results

async def handle_get_batch_server_status(request: BatchServerPortRequest) -> List[ServerStatusResponse]: # Added return type hint
    results = []
    logger.info(f"Batch server status request received for ports: {request.ports}")
    for port_num in request.ports:
        try:
            status_resp = await handle_get_server_status(port_num)
            results.append(status_resp)
        except HTTPException as e: # Capture HTTPExceptions from handle_get_server_status (e.g. 404)
             # Find config if it exists even if server is down
            cfg_data = get_server_configs().get(("127.0.0.1", port_num))
            cfg_object = ServerConfig(**cfg_data) if cfg_data else None
            results.append(ServerStatusResponse(
                port=port_num, 
                status="error" if e.status_code >=500 else "not_found", # or map status code
                detail=e.detail,
                config=cfg_object
            ))
        except Exception as e: # Catch any other unexpected error
            results.append(ServerStatusResponse(port=port_num, status="error", detail=str(e)))
    return results

async def handle_get_all_server_statuses() -> List[ServerStatusResponse]: # Added return type hint
    results = []
    logger.info("Request received for status of all server instances.")
    current_server_processes = get_server_processes()
    current_server_configs = get_server_configs()
    known_ports = set(key[1] for key in current_server_processes.keys()) | set(key[1] for key in current_server_configs.keys())

    if not known_ports: return []
    sorted_ports = sorted(list(known_ports))

    for port_num in sorted_ports:
        try:
            status_resp = await handle_get_server_status(port_num)
            results.append(status_resp)
        except HTTPException as e:
            cfg_data = current_server_configs.get(("127.0.0.1", port_num))
            cfg_object = ServerConfig(**cfg_data) if cfg_data else None
            results.append(ServerStatusResponse(
                port=port_num, 
                status="error" if e.status_code >=500 else "not_found", # or map status code
                detail=e.detail,
                config=cfg_object
            ))
        except Exception as e: # Catch any other unexpected error
             results.append(ServerStatusResponse(port=port_num, status="error", detail=str(e)))
    return results

async def handle_terminate_all_servers() -> List[ServerTerminationResponse]: # New function
    """Terminates all tracked llama-server instances."""
    results = []
    host = "127.0.0.1"
    logger.info("Request received to terminate all server instances.")
    
    current_server_processes = get_server_processes()
    # Get ports of currently active processes only
    active_ports = [key[1] for key, proc_entry in current_server_processes.items() if proc_entry.get("process") and proc_entry["process"].returncode is None]

    if not active_ports:
        logger.info("No active server instances found to terminate.")
        return []

    logger.info(f"Attempting to terminate servers on ports: {active_ports}")
    for port_num in active_ports:
        key = (host, port_num)
        message = await _terminate_server_process(key) # _terminate_server_process handles cleanup
        status_str = "unknown"
        if "terminated successfully" in message or "forcefully killed" in message: status_str = "terminated"
        elif "already stopped" in message: status_str = "already_stopped" # Should not happen if we only list active
        elif "No server process found" in message: status_str = "not_found" # Should not happen
        elif "Error" in message: status_str = "error_termination_failed"
        results.append(ServerTerminationResponse(port=port_num, message=message, status=status_str))
    
    # Optionally, could add a pass for ports in configs but not in active processes to mark them as 'already_stopped'
    # For now, only acts on explicitly active processes.
    
    logger.info(f"Termination attempt for all servers completed. Results: {results}")
    return results
