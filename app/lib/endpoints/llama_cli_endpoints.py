import os
import logging
import asyncio
import shlex
from fastapi import HTTPException, status
from typing import Dict, Any, List

from ..models import (
    LlamaCliInitRequest,
    LlamaCliChatRequest,
    BatchLlamaCliInitRequest,
    BatchLlamaCliRemoveRequest,
    BatchLlamaCliChatRequest,
    BatchLlamaCliStatusRequest
)

# Import the process management functions for persistent sessions
from .process_management import (
    start_cli_chat_process,
    send_to_cli_chat_session,
    terminate_cli_chat_session,
    cli_chat_sessions # Direct access for status checks
)

logger = logging.getLogger(__name__)

# This model path is used for all CLI sessions.
STATIC_MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

async def initialize_llama_cli_session(request: LlamaCliInitRequest) -> Dict[str, Any]:
    """
    Starts a persistent llama-cli process in conversational mode.
    The cli_alias from the request is used as the unique session_id.
    """
    session_id = request.cli_alias
    if session_id in cli_chat_sessions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A CLI chat session with alias '{session_id}' is already running."
        )

    try:
        # Start the persistent process
        session_data = await start_cli_chat_process(
            session_id=session_id,
            model_path=STATIC_MODEL_PATH,
            threads=request.threads,
            ctx_size=request.ctx_size,
            n_predict=request.n_predict,
            temperature=request.temperature,
            repeat_penalty=request.repeat_penalty,
            top_k=request.top_k,
            top_p=request.top_p,
            system_prompt=request.system_prompt,
        )
        logger.info(f"Successfully started persistent llama-cli session '{session_id}' (PID: {session_data['pid']}).")
        return {
            "cli_alias": session_id,
            "status": "running",
            "pid": session_data["pid"],
            "message": "CLI process started successfully in conversational mode."
        }
    except FileNotFoundError:
        logger.error(f"Failed to start CLI session '{session_id}': llama-cli executable not found.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Llama-cli executable not found. Please ensure it's in your PATH or the LLAMA_CLI_PATH environment variable is set correctly."
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to start persistent CLI session '{session_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while starting the CLI process: {str(e)}"
        )

async def chat_with_llama_cli_session(chat_request: LlamaCliChatRequest) -> Dict[str, Any]:
    """
    Sends a prompt to a running persistent llama-cli session.
    """
    session_id = chat_request.cli_alias
    prompt = chat_request.prompt

    try:
        response_text = await send_to_cli_chat_session(session_id, prompt)
        return {
            "cli_alias": session_id,
            "prompt": prompt,
            "response": response_text
        }
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (IOError, TimeoutError) as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during chat with session '{session_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

async def shutdown_llama_cli_session(cli_alias: str) -> Dict[str, str]:
    """
    Terminates a persistent llama-cli process.
    """
    try:
        message = await terminate_cli_chat_session(cli_alias)
        logger.info(f"Termination command for session '{cli_alias}' processed. Result: {message}")
        return {"cli_alias": cli_alias, "status": "terminated", "message": message}
    except Exception as e:
        logger.error(f"Failed to terminate CLI session '{cli_alias}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during termination: {str(e)}"
        )

async def get_llama_cli_session_status(cli_alias: str) -> Dict[str, Any]:
    """
    Retrieves the status of a specific persistent llama-cli session.
    """
    session_info = cli_chat_sessions.get(cli_alias)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active CLI chat session found with alias '{cli_alias}'."
        )

    process = session_info.get("process")
    status = "stopped"
    if process and process.returncode is None:
        status = "running"

    # Return a safe subset of the session data
    return {
        "cli_alias": cli_alias,
        "status": status,
        "pid": session_info.get("pid"),
        "model_path": session_info.get("model_path"),
        "start_time": session_info.get("start_time"),
        "last_interaction_time": session_info.get("last_interaction_time"),
        "command": " ".join(session_info.get("command", []))
    }

# --- Batch Operations ---
async def handle_initialize_batch_llama_cli_configs(batch_request: BatchLlamaCliInitRequest) -> List[Dict[str, Any]]:
    """
    Processes a batch request to start multiple persistent llama-cli sessions.
    """
    aliases = [req.cli_alias for req in batch_request.requests]
    if len(aliases) != len(set(aliases)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duplicate cli_alias values found in the batch request."
        )

    async def process_request(req: LlamaCliInitRequest):
        try:
            result = await initialize_llama_cli_session(req)
            return {"cli_alias": req.cli_alias, "status": "success", "data": result}
        except HTTPException as e:
            return {"cli_alias": req.cli_alias, "status": "error", "detail": e.detail, "status_code": e.status_code}
        except Exception as e:
            logger.error(f"Unexpected error processing batch init for alias {req.cli_alias}: {str(e)}", exc_info=True)
            return {"cli_alias": req.cli_alias, "status": "error", "detail": "An unexpected server error occurred.", "status_code": 500}

    results = await asyncio.gather(*(process_request(req) for req in batch_request.requests))
    return results

async def handle_remove_batch_llama_cli_configs(batch_request: BatchLlamaCliRemoveRequest) -> List[Dict[str, Any]]:
    """
    Processes a batch request to terminate multiple persistent llama-cli sessions.
    """
    async def process_request(alias: str):
        try:
            result = await shutdown_llama_cli_session(alias)
            return {"cli_alias": alias, "status": "success", "data": result}
        except HTTPException as e:
            return {"cli_alias": alias, "status": "error", "detail": e.detail, "status_code": e.status_code}
        except Exception as e:
            logger.error(f"Unexpected error processing batch removal for alias {alias}: {str(e)}", exc_info=True)
            return {"cli_alias": alias, "status": "error", "detail": "An unexpected server error occurred.", "status_code": 500}

    results = await asyncio.gather(*(process_request(alias) for alias in batch_request.aliases))
    return results

async def handle_batch_chat_with_llama_cli(batch_request: BatchLlamaCliChatRequest) -> List[Dict[str, Any]]:
    """
    Processes a batch of chat requests with their respective llama-cli sessions.
    """
    async def process_request(req: LlamaCliChatRequest):
        try:
            # Reuse the single chat handler logic
            result = await chat_with_llama_cli_session(req)
            return {"cli_alias": req.cli_alias, "status": "success", "data": result}
        except HTTPException as e:
            return {"cli_alias": req.cli_alias, "status": "error", "detail": e.detail, "status_code": e.status_code}
        except Exception as e:
            logger.error(f"Unexpected error processing batch chat for alias {req.cli_alias}: {str(e)}", exc_info=True)
            return {"cli_alias": req.cli_alias, "status": "error", "detail": "An unexpected server error occurred.", "status_code": 500}

    results = await asyncio.gather(*(process_request(req) for req in batch_request.requests))
    return results

async def handle_get_batch_llama_cli_status(batch_request: BatchLlamaCliStatusRequest) -> List[Dict[str, Any]]:
    """
    Processes a batch request to get the status of multiple persistent llama-cli sessions.
    """
    async def process_request(alias: str):
        try:
            # Reuse the single status handler logic
            result = await get_llama_cli_session_status(alias)
            return {"cli_alias": alias, "status": "success", "data": result}
        except HTTPException as e:
            return {"cli_alias": alias, "status": "error", "detail": e.detail, "status_code": e.status_code}
        except Exception as e:
            logger.error(f"Unexpected error processing batch status for alias {alias}: {str(e)}", exc_info=True)
            return {"cli_alias": alias, "status": "error", "detail": "An unexpected server error occurred.", "status_code": 500}

    results = await asyncio.gather(*(process_request(alias) for alias in batch_request.aliases))
    return results
