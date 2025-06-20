import psutil
import os
import asyncio
import logging
import atexit
from typing import Dict, Tuple, Any, List # Added List
import uuid # Added for session IDs
from datetime import datetime, timezone # Added for timestamps
import asyncio.subprocess # Added for explicit subprocess typing

logger = logging.getLogger(__name__)

server_processes: Dict[Tuple[str, int], Dict[str, Any]] = {}
server_configs: Dict[Tuple[str, int], Dict[str, Any]] = {}

# New dictionary for CLI chat sessions
cli_chat_sessions: Dict[str, Dict[str, Any]] = {}

def get_server_processes():
    return server_processes

def get_server_configs():
    return server_configs

# New getter for CLI chat sessions
def get_cli_chat_sessions():
    return cli_chat_sessions

def _max_server_instances_by_ram(per_server_gb=1):
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    used_gb = psutil.virtual_memory().used / (1024 ** 3)
    available_gb = total_gb - used_gb
    return int(available_gb // per_server_gb)

FASTAPI_PORT = 8080 # Assuming this is defined, might need to get from config or env
build_dir = os.getenv("BUILD_DIR", "build")
LLAMA_CLI_EXECUTABLE = os.path.join(build_dir, "bin", "llama-cli")

CLI_RESPONSE_TIMEOUT = float(os.getenv("CLI_RESPONSE_TIMEOUT", "60.0")) # Timeout for CLI response
CLI_USER_PROMPT_MARKER = os.getenv("CLI_USER_PROMPT_MARKER", "> ") # How llama-cli prompts for user input

_atexit_cleanup_completed = False

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

async def start_cli_chat_process(
    session_id: str,
    model_path: str,
    threads: int,
    ctx_size: int,
    n_predict: int,
    temperature: float,
    repeat_penalty: float,
    top_k: int,
    top_p: float,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    """
    Starts a llama-cli process in conversational mode (-cnv).
    """
    if session_id in cli_chat_sessions:
        logger.warning(f"CLI chat session {session_id} already exists.")
        raise ValueError(f"CLI chat session {session_id} already exists.")

    command = [
        LLAMA_CLI_EXECUTABLE,
        '-m', model_path,
        '-t', str(threads),
        '-c', str(ctx_size),
        '-n', str(n_predict),
    ]

    # Only add optional arguments if they are not None or blank
    if temperature is not None and str(temperature) != '':
        command.extend(['--temp', str(temperature)])
    if repeat_penalty is not None and str(repeat_penalty) != '':
        command.extend(['--repeat-penalty', str(repeat_penalty)])
    if top_k is not None and str(top_k) != '':
        command.extend(['--top-k', str(top_k)])
    if top_p is not None and str(top_p) != '':
        command.extend(['--top-p', str(top_p)])

    if system_prompt:
        command.extend(['-p', system_prompt])

    command.append('-cnv')

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Robust startup: Wait for the process to be ready or fail.
        try:
            # We will read stdout in chunks to avoid getting stuck on readline()
            # if the prompt marker doesn't end with a newline.
            # The entire readiness check is wrapped in a single timeout.
            async def _wait_for_readiness():
                output_buffer = b""
                while True:
                    # Read a small chunk to avoid blocking.
                    chunk = await process.stdout.read(128)
                    if chunk:
                        logger.debug(f"Read chunk from stdout for session '{session_id}': {chunk!r}")
                    else:
                        logger.warning(f"Read empty chunk from stdout for session '{session_id}'. Process may have exited.")

                    if not chunk:
                        # If the process exits, read stderr to find out why.
                        stderr_output = await process.stderr.read()
                        error_details = stderr_output.decode(errors='replace').strip()
                        logger.error(f"Llama-cli process for '{session_id}' exited during startup. Stderr: {error_details}")
                        raise RuntimeError(f"The llama-cli process exited unexpectedly during startup. Stderr: {error_details}")
                    
                    output_buffer += chunk
                    decoded_output = output_buffer.decode(errors='replace')
                    
                    # Check if the prompt marker has appeared in the output.
                    if CLI_USER_PROMPT_MARKER.strip() in decoded_output:
                        logger.info(f"CLI session {session_id} is ready. Startup output captured.")
                        logger.debug(f"Startup output from {session_id}: {decoded_output}")
                        return # Success
            
            await asyncio.wait_for(_wait_for_readiness(), timeout=180.0)

        except (asyncio.TimeoutError, RuntimeError) as e:
            # If the process fails to become ready, kill it and report the error.
            logger.error(f"The llama-cli process for session '{session_id}' failed to become ready. Exception: {repr(e)}", exc_info=True)
            if process.returncode is None:
                logger.info(f"Killing process {process.pid} for session '{session_id}' as it failed to start.")
                process.kill()
            await process.wait()
            logger.info(f"Reading stderr from failed process {process.pid} for session '{session_id}'.")
            stderr_output = await process.stderr.read()
            error_details = stderr_output.decode(errors='replace').strip()
            logger.error(f"Stderr from failed llama-cli process '{session_id}': {error_details or '<no stderr output>'}")
            raise RuntimeError(f"Failed to start llama-cli process. Details: {error_details or 'No error output.'}")

        session_data = {
            "process": process,
            "stdin": process.stdin,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "pid": process.pid,
            "status": "running",
            "model_path": model_path,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "last_interaction_time": datetime.now(timezone.utc).isoformat(),
        }
        cli_chat_sessions[session_id] = session_data
        logger.info(f"Started CLI chat session {session_id} (PID: {process.pid}) with model {model_path}.")
        logger.debug(f"CLI command for session {session_id}: {' '.join(command)}")
        return session_data
    except FileNotFoundError:
        logger.error(f"Failed to start CLI chat session {session_id}: {LLAMA_CLI_EXECUTABLE} not found. Ensure it's in PATH or build was successful.")
        raise
    except Exception as e:
        logger.error(f"An unhandled exception occurred when trying to start CLI chat session {session_id}: {e}", exc_info=True)
        raise

async def send_to_cli_chat_session(session_id: str, prompt: str) -> str:
    """
    Sends a prompt to an active CLI chat session and gets the response.
    """
    session_entry = cli_chat_sessions.get(session_id)
    if not session_entry or session_entry["process"].returncode is not None:
        logger.error(f"CLI chat session {session_id} not found or process terminated.")
        raise LookupError(f"CLI chat session {session_id} not active.")

    process: asyncio.subprocess.Process = session_entry["process"]
    stdin_writer: asyncio.StreamWriter = session_entry["stdin"]
    stdout_reader: asyncio.StreamReader = session_entry["stdout"]

    try:
        if stdin_writer.is_closing():
            logger.error(f"Stdin for CLI session {session_id} is closed.")
            raise IOError(f"Cannot write to closed stdin for session {session_id}.")

        logger.debug(f"Sending to CLI session {session_id} (PID: {process.pid}): '{prompt}'")
        stdin_writer.write(prompt.encode() + b'\n')
        await stdin_writer.drain()

        async def _read_response():
            response_buffer = b""
            while True:
                chunk = await stdout_reader.read(128)
                if not chunk:
                    raise IOError("Process exited while waiting for response.")

                response_buffer += chunk
                decoded_response = response_buffer.decode(errors='replace')
                
                # Check if the response is complete by looking for the next prompt marker.
                if CLI_USER_PROMPT_MARKER.strip() in decoded_response:
                    # The actual response is everything *before* the prompt marker.
                    response_text, _, _ = decoded_response.partition(CLI_USER_PROMPT_MARKER.strip())
                    return response_text.strip()

        # Wrap the entire read operation in the timeout.
        raw_response = await asyncio.wait_for(_read_response(), timeout=CLI_RESPONSE_TIMEOUT)
        
        # Clean the response: llama-cli may echo the prompt, so we remove it.
        # We compare based on the stripped prompt to be robust against whitespace differences.
        cleaned_response = raw_response
        prompt_to_check = prompt.strip()
        if raw_response.strip().startswith(prompt_to_check):
            # Use removeprefix for a safe, non-overlapping removal of the prefix.
            # lstrip() removes any leading whitespace/newlines after the prompt.
            cleaned_response = raw_response.strip().removeprefix(prompt_to_check).lstrip()

        session_entry["last_interaction_time"] = datetime.now(timezone.utc).isoformat()
        logger.debug(f"Raw response from CLI session {session_id}: '{raw_response}'")
        logger.debug(f"Cleaned response for CLI session {session_id}: '{cleaned_response}'")
        return cleaned_response

    except asyncio.TimeoutError:
        logger.error(f"Timeout receiving response from CLI session {session_id}.")
        raise TimeoutError(f"Timeout receiving response from CLI session {session_id}.")
    except (BrokenPipeError, ConnectionResetError, IOError) as e:
        logger.error(f"Pipe broken or connection reset for CLI session {session_id} (PID: {process.pid}): {e}. Process likely crashed or exited.", exc_info=True)
        await terminate_cli_chat_session(session_id) # Attempt cleanup
        raise IOError(f"Communication error with CLI session {session_id}: {e}")
    except Exception as e:
        logger.error(f"Error communicating with CLI session {session_id} (PID: {process.pid}): {e}", exc_info=True)
        raise

async def terminate_cli_chat_session(session_id: str) -> str:
    """
    Terminates a specific CLI chat session.
    """
    session_entry = cli_chat_sessions.pop(session_id, None)
    if not session_entry:
        logger.warning(f"Attempted to terminate non-existent CLI chat session {session_id}.")
        return f"CLI chat session {session_id} not found."

    process: asyncio.subprocess.Process = session_entry["process"]
    pid = session_entry["pid"]
    status_message = ""

    if process.returncode is None: # If still running
        logger.info(f"Attempting to terminate CLI chat session {session_id} (PID: {pid}).")
        try:
            # llama-cli -cnv might not exit cleanly with SIGTERM if it's waiting for input.
            # Sending a newline or 'exit' command first might be more graceful if supported.
            if process.stdin and not process.stdin.is_closing():
                try:
                    process.stdin.write(b'exit\\n') # Try a graceful exit command
                    await process.stdin.drain()
                    await asyncio.wait_for(process.wait(), timeout=2.0) # Short wait for graceful exit
                    logger.info(f"CLI chat session {session_id} (PID: {pid}) exited gracefully after 'exit' command.")
                except (asyncio.TimeoutError, BrokenPipeError, AttributeError): # AttributeError if stdin is None
                    logger.warning(f"CLI session {session_id} did not exit gracefully or stdin closed, proceeding with terminate.")
                    process.terminate() # Send SIGTERM
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    logger.info(f"CLI chat session {session_id} (PID: {pid}) terminated successfully after SIGTERM.")
            else: # stdin not available or closed
                process.terminate() # Send SIGTERM
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"CLI chat session {session_id} (PID: {pid}) terminated successfully after SIGTERM.")
            status_message = f"CLI chat session {session_id} (PID: {pid}) terminated."
        except asyncio.TimeoutError:
            logger.warning(f"CLI chat session {session_id} (PID: {pid}) did not respond to SIGTERM/exit. Attempting SIGKILL.")
            try:
                process.kill() # Send SIGKILL
                await process.wait() # Ensure it's killed
                logger.info(f"CLI chat session {session_id} (PID: {pid}) forcefully killed.")
                status_message = f"CLI chat session {session_id} (PID: {pid}) forcefully killed."
            except Exception as e_kill:
                logger.error(f"Error forcefully killing CLI session {session_id} (PID: {pid}): {str(e_kill)}", exc_info=True)
                status_message = f"Error forcefully killing CLI session {session_id} (PID: {pid}): {str(e_kill)}. Process may still be running."
        except Exception as e_term:
            logger.error(f"Error terminating CLI session {session_id} (PID: {pid}): {str(e_term)}", exc_info=True)
            status_message = f"Error terminating CLI session {session_id} (PID: {pid}): {str(e_term)}. Process may still be running."
    else:
        logger.info(f"CLI chat session {session_id} (PID: {pid}) was already stopped (return code: {process.returncode}).")
        status_message = f"CLI chat session {session_id} (PID: {pid}) was already stopped."
    
    # Ensure stdin/stdout/stderr are closed if process object still exists
    if process.stdin and not process.stdin.is_closing():
        process.stdin.close()
    # stdout/stderr readers are typically closed when the process exits or by reading EOF.

    return status_message

async def _terminate_all_managed_processes(): # Renamed and will be updated
    global _atexit_cleanup_completed
    if _atexit_cleanup_completed:
        return
    
    logger.info("Attempting to terminate all managed processes asynchronously at exit.")
    
    # Terminate server processes
    server_keys_to_terminate = list(server_processes.keys())
    server_tasks = [_terminate_server_process(key) for key in server_keys_to_terminate]
    
    # Terminate CLI chat sessions
    cli_session_ids_to_terminate = list(cli_chat_sessions.keys())
    cli_tasks = [terminate_cli_chat_session(session_id) for session_id in cli_session_ids_to_terminate]
    
    all_tasks = server_tasks + cli_tasks
    all_keys = server_keys_to_terminate + cli_session_ids_to_terminate # For logging

    if not all_tasks:
        logger.info("No managed processes to terminate at exit.")
        _atexit_cleanup_completed = True
        return

    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    for i, key_or_id in enumerate(all_keys):
        result = results[i]
        process_type = "server" if i < len(server_keys_to_terminate) else "CLI session"
        if isinstance(result, Exception):
            logger.error(f"Error during atexit termination for {process_type} {key_or_id}: {result}", exc_info=result)
        else:
            logger.info(f"Atexit termination for {process_type} {key_or_id}: {result}")
            
    logger.info("Asynchronous termination of all managed processes at exit completed.")
    _atexit_cleanup_completed = True

def _run_async_cleanup_on_exit():
    # Ensure this function uses the new combined termination function
    try:
        # Check if an event loop is running, if not, create one for cleanup.
        # This handles cases where atexit is called after the main loop is closed.
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
            loop.create_task(_terminate_all_managed_processes()) # Schedule as task if loop is running
        except RuntimeError: # No running loop or loop is closed
            asyncio.run(_terminate_all_managed_processes())
            
    except RuntimeError as e:
        if ("cannot schedule new futures after shutdown" in str(e).lower() or 
            "event loop is closed" in str(e).lower() or
            "no running event loop" in str(e).lower()): # Added "no running event loop"
            logger.warning(f"Could not run async cleanup at exit because event loop was closed, shutting down, or not available: {e}")
        else:
            logger.error(f"Unexpected RuntimeError during atexit async cleanup: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected Exception during atexit async cleanup: {e}", exc_info=True)

# Ensure the new function name is registered if it was renamed
# atexit.unregister(_run_async_cleanup_on_exit) # If previous was different
atexit.register(_run_async_cleanup_on_exit) # Register the (potentially updated) cleanup
