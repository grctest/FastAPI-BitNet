import psutil
import os
import asyncio
import logging
import atexit
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

server_processes: Dict[Tuple[str, int], Dict[str, Any]] = {}
server_configs: Dict[Tuple[str, int], Dict[str, Any]] = {}

def get_server_processes():
    return server_processes

def get_server_configs():
    return server_configs
FASTAPI_PORT = 8080
_atexit_cleanup_completed = False

def _max_server_instances_by_ram(per_server_gb=1):
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    used_gb = psutil.virtual_memory().used / (1024 ** 3)
    available_gb = total_gb - used_gb
    return int(available_gb // per_server_gb)

async def _terminate_server_process(key: tuple[str, int]):
    host, port = key
    if port == FASTAPI_PORT:
        logger.warning(f"Attempt to terminate FastAPI server on port {port} denied.")
        return f"Operation denied: Port {port} is used by the FastAPI application and cannot be terminated via this endpoint."
    proc_entry = server_processes.get(key)
    if not proc_entry:
        server_configs.pop(key, None)
        logger.info(f"No server process found for key {key} (port {port}) during termination attempt.")
        return f"No server process found for key {key} (port {port})."
    proc_to_terminate = proc_entry["process"]
    pid = proc_entry["pid"]
    if proc_to_terminate.returncode is None:
        logger.info(f"Attempting to terminate server on port {port} (PID: {pid}).")
        try:
            proc_to_terminate.terminate()
            await asyncio.wait_for(proc_to_terminate.wait(), timeout=5.0)
            logger.info(f"Server on port {port} (PID: {pid}) terminated successfully after SIGTERM.")
            server_processes.pop(key, None)
            server_configs.pop(key, None)
            return f"Server on port {port} (PID: {pid}) terminated successfully."
        except asyncio.TimeoutError:
            logger.warning(f"Server on port {port} (PID: {pid}) did not respond to SIGTERM within timeout. Attempting SIGKILL.")
            try:
                proc_to_terminate.kill()
                await proc_to_terminate.wait()
                logger.info(f"Server on port {port} (PID: {pid}) forcefully killed.")
                server_processes.pop(key, None)
                server_configs.pop(key, None)
                return f"Server on port {port} (PID: {pid}) forcefully killed as it did not respond to SIGTERM."
            except Exception as e_kill:
                logger.error(f"Error forcefully killing server on port {port} (PID: {pid}): {str(e_kill)}", exc_info=True)
                return f"Error forcefully killing server on port {port} (PID: {pid}): {str(e_kill)}. Process may still be running."
        except Exception as e_term:
            logger.error(f"Error terminating server on port {port} (PID: {pid}) with SIGTERM: {str(e_term)}", exc_info=True)
            return f"Error terminating server on port {port} (PID: {pid}): {str(e_term)}. Process may still be running."
    else:
        logger.info(f"Server on port {port} was already stopped (return code: {proc_to_terminate.returncode}). Cleaned up tracking.")
        server_processes.pop(key, None)
        server_configs.pop(key, None)
        return f"Server on port {port} was already stopped. Cleaned up tracking."

async def _terminate_all_servers():
    global _atexit_cleanup_completed
    if _atexit_cleanup_completed:
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
        asyncio.run(_terminate_all_servers())
    except RuntimeError as e:
        if ("cannot schedule new futures after shutdown" in str(e).lower() or "event loop is closed" in str(e).lower()):
            logger.warning(f"Could not run async cleanup at exit because event loop was closed or shutting down: {e}")
        else:
            logger.error(f"Unexpected RuntimeError during atexit async cleanup: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected Exception during atexit async cleanup: {e}", exc_info=True)

atexit.register(_run_async_cleanup_on_exit)
