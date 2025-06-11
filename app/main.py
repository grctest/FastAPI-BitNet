import psutil
import os
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi_mcp import FastApiMCP

from lib.models import ModelEnum
from lib.endpoints.chat_endpoints import ChatRequest, MultiChatRequest
from lib.endpoints.chat_endpoints import chat_with_bitnet, multichat_with_bitnet
from lib.endpoints.server_endpoints import BatchServerInitRequest, BatchServerPortRequest
from lib.endpoints.server_endpoints import initialize_server_endpoint, initialize_batch_servers_endpoint, shutdown_server_endpoint, shutdown_batch_servers_endpoint, get_server_status, get_batch_server_status_endpoint
from lib.endpoints.benchmark_endpoints import run_benchmark, run_perplexity, get_model_sizes

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Adjust level as needed (e.g., logging.DEBUG for more detail)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# --- FastAPI Application ---
app = FastAPI(
    title="FastAPI BitNet Orchestrator",
    description="An API for managing and interacting with BitNet server instances (based on llama.cpp). \\nProvides endpoints for initializing, shutting down, and checking the status of individual and batch BitNet servers, \\nrunning benchmarks, calculating perplexity, and performing chat completions.",
    version="0.1.0",
    contact={
        "name": "Project Mantainers",
        "url": "https://github.com/grctest/FastAPI-BitNet",
    },
    license_info={
        "name": "MIT License", # Or your chosen license
        "url": "https://opensource.org/licenses/MIT", # Link to license
    },
)

@app.get(
    "/estimate",
    summary="Estimate max BitNet servers by RAM and CPU threads",
    tags=["Server Management"]
)
async def bitnet_server_capacity(per_server_gb: float = 1.5):
    """
    Estimate the maximum number of BitNet server instances that can be run on this machine.

    This endpoint provides a quick estimate for the AI or user to determine how many 1-thread BitNet servers can be launched,
    based on two system resource constraints:
    
    1. **Available RAM**: Calculates the number of servers that can be run given the currently available RAM (default 1GB per server).
    2. **CPU Threads**: Calculates the number of servers that could be run if each uses a single CPU thread (based on total logical CPUs).

    Args:
        per_server_gb (float, optional): Estimated RAM usage per server in GB. Defaults to 1.5.

    Returns:
        dict: {
            "max_1thread_servers_by_available_ram": int,  # Max servers by available RAM
            "available_ram_gb": float,                    # Currently available RAM in GB
            "total_ram_gb": float,                        # Total system RAM in GB
            "max_1thread_servers_by_cpu_threads": int,    # Max servers by total CPU threads
            "cpu_threads": int,                           # Number of logical CPU threads
            "note": str                                   # Guidance for users/AI
        }
    """
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    max_ram_servers = int(available_ram_gb // per_server_gb)
    cpu_threads = os.cpu_count() or 1
    max_cpu_servers = cpu_threads
    return {
        "max_1thread_servers_by_available_ram": max_ram_servers,
        "available_ram_gb": round(available_ram_gb, 2),
        "total_ram_gb": round(total_ram_gb, 2),
        "max_1thread_servers_by_cpu_threads": max_cpu_servers,
        "cpu_threads": cpu_threads,
        "note": (
            "You can run up to the minimum of these two values. "
            "Freeing up RAM may increase the RAM-based limit. "
            "CPU thread count is a hardware maximum, not current usage."
        )
    }

@app.post("/initialize-server", summary="Initialize a Single BitNet Server", tags=["Server Management"])
async def initialize_server(
    threads: int = Query(os.cpu_count() // 2, gt=0, le=os.cpu_count(), description="Number of threads for the server instance. Must be > 0 and <= system CPU count."),
    ctx_size: int = Query(2048, gt=0, description="Context size (in tokens) for the server instance. Must be > 0."),
    port: int = Query(8081, gt=1023, description="Port for the server instance. Must be > 1023 and unique."),
    system_prompt: str = Query("You are a helpful assistant.", description="System prompt for the language model instance."),
    n_predict: int = Query(4096, gt=0, description="Maximum number of tokens to predict. Must be > 0."),
    temperature: float = Query(0.8, gt=0.0, le=2.0, description="Sampling temperature (0.0 - 2.0). Higher values make output more random.")
):
    """
    Initializes and starts a single BitNet (llama-server) process in the background.

    This endpoint allows configuring a new server instance with specific threading, context size,
    port, system prompt, prediction length, and temperature. It checks for port conflicts and
    ensures that the requested threads do not exceed system capacity or lead to oversubscription
    when considering already running instances.

    **Parameters**:
    - `threads`: Number of CPU threads for the server.
    - `ctx_size`: Context window size for the model.
    - `port`: Network port for the server. Must be unique and > 1023.
    - `system_prompt`: The initial system message to guide the model's behavior.
    - `n_predict`: Default maximum number of tokens the server will generate in a response.
    - `temperature`: Sampling temperature for generation.

    **Successful Response (200 OK)**:
    - JSON object containing a success message, the PID of the started server process,
      and the server's configuration.
      Example: `{"message": "Server started on 127.0.0.1:8082", "pid": 12345, "config": {...}}`

    **Error Responses**:
    - `400 Bad Request`: If requested threads exceed system capacity.
    - `409 Conflict`: If a server is already running on the specified port.
    - `429 Too Many Requests`: If starting the server would oversubscribe CPU threads.
    - `500 Internal Server Error`: If the server binary is not found or fails to start for other reasons.
    """
    try:
        return await initialize_server_endpoint(
            threads=threads,
            ctx_size=ctx_size,
            port=port,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Error in /initialize-server endpoint: {str(e)}", exc_info=True)
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

@app.post("/shutdown-server", summary="Shutdown a Single BitNet Server", tags=["Server Management"])
async def shutdown_server(port: int = Query(..., gt=1023, description="The port of the BitNet server instance to shut down.")):
    """
    Shuts down a specific BitNet server instance managed by this orchestrator.

    Identifies the server by its port number and attempts to terminate the process.
    It will first try a graceful termination (SIGTERM) and then a forceful kill (SIGKILL)
    if the server does not respond.

    **Parameters**:
    - `port`: The network port of the server instance to be shut down.

    **Successful Response (200 OK)**:
    - JSON object with a message indicating the outcome of the shutdown attempt.
      Example: `{"message": "Server on port 8082 (PID: 12345) terminated successfully."}`
      Example: `{"message": "Server on port 8082 was already stopped. Cleaned up tracking."}`


    **Error Responses**:
    - `403 Forbidden`: If attempting to shut down the FastAPI orchestrator itself.
    - `404 Not Found`: If no server is found running on the specified port or managed by the orchestrator.
    - `500 Internal Server Error`: If an error occurs during the termination process.
    """
    try:
        return await shutdown_server_endpoint(port=port)
    except HTTPException as e: # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        logger.error(f"Error in /shutdown-server endpoint for port {port}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/server-status", summary="Get Status of a Single BitNet Server", tags=["Server Management"])
async def server_status_endpoint(port: int = Query(..., gt=1023, description="The port of the BitNet server instance to check.")):
    """
    Retrieves the status of a specific BitNet server instance.

    Checks if a server process is running on the given port and is managed by this orchestrator.
    Returns the server's status (running/stopped), PID (if running), and its configuration.

    **Parameters**:
    - `port`: The network port of the server instance.

    **Successful Response (200 OK)**:
    - JSON object detailing the server's status.
      If running: `{"status": "running", "pid": 12345, "config": {...}}`
      If stopped: `{"status": "stopped", "config": {...}}` or `{"status": "stopped", "config": "No configuration found for port XXXX."}`
    """
    return await get_server_status(port=port) # Call the function from endpoints.py

# --- Batch Server Endpoints ---
@app.post("/initialize-batch-servers", summary="Initialize Multiple BitNet Servers", tags=["Batch Server Management"])
async def initialize_batch_servers(request: BatchServerInitRequest):
    """
    Initializes and starts multiple BitNet server instances based on a list of configurations.

    This endpoint processes a batch request to start several servers. It performs batch-level
    validations for duplicate ports and total thread capacity before attempting to start
    each server. Each server initialization is handled individually, and results for each
    are returned.

    **Request Body**:
    - `BatchServerInitRequest`: A JSON object containing a list of `servers`. Each item in the
      `servers` list defines the configuration for one server instance (threads, ctx_size, port, etc.).
      See `ServerInitConfig` model for details.

    **Successful Response (200 OK)**:
    - A JSON list where each item corresponds to a server in the request, detailing the
      outcome of its initialization attempt (success or error, message, PID if successful, config).
      Example: `[{"port": 8082, "status": "success", "message": "...", "pid": 12345, "config": {...}}, {"port": 8083, "status": "error", "message": "..."}]`

    **Error Responses (for batch-level validation failures)**:
    - `400 Bad Request`: If duplicate ports are found within the batch request.
    - `429 Too Many Requests`: If the total requested threads for the batch exceed system capacity or
      available threads (considering already running instances).
    - `500 Internal Server Error`: If the server binary is not found, or an unexpected error occurs
      during the batch processing logic itself. Individual server start failures are reported in the
      200 OK response list.
    """
    try:
        return await initialize_batch_servers_endpoint(request=request)
    except HTTPException as e: # Catch HTTPExceptions specifically to re-raise them
        raise e
    except Exception as e:
        logger.error("Error in /initialize-batch-servers endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch server initialization: {str(e)}")

@app.post("/shutdown-batch-servers", summary="Shutdown Multiple BitNet Servers", tags=["Batch Server Management"])
async def shutdown_batch_servers(request: BatchServerPortRequest):
    """
    Shuts down multiple BitNet server instances based on a list of port numbers.

    Processes a batch request to terminate several servers. Each server shutdown is attempted
    individually.

    **Request Body**:
    - `BatchServerPortRequest`: A JSON object containing a list of `ports` (integers)
      of the server instances to be shut down.

    **Successful Response (200 OK)**:
    - A JSON list where each item corresponds to a port in the request, detailing the
      outcome of its shutdown attempt (e.g., success_terminated, denied, not_found, error_termination_failed).
      Example: `[{"port": 8082, "status": "success_terminated", "message": "..."}, {"port": 8083, "status": "not_found", "message": "..."}]`

    **Error Responses**:
    - `500 Internal Server Error`: If an unexpected error occurs during the batch processing logic.
      Individual server shutdown failures/statuses are reported in the 200 OK response list.
    """
    try:
        return await shutdown_batch_servers_endpoint(request=request)
    except Exception as e:
        logger.error("Error in /shutdown-batch-servers endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch server shutdown: {str(e)}")

@app.post("/batch-server-status", summary="Get Status of Multiple BitNet Servers", tags=["Batch Server Management"])
async def batch_server_status(request: BatchServerPortRequest):
    """
    Retrieves the status of multiple BitNet server instances based on a list of port numbers.

    **Request Body**:
    - `BatchServerPortRequest`: A JSON object containing a list of `ports` (integers)
      of the server instances to check.

    **Successful Response (200 OK)**:
    - A JSON list where each item corresponds to a port in the request, detailing its
      status (running/stopped), PID (if running), and configuration.
      Example: `[{"port": 8082, "status": "running", "pid": 12345, "config": {...}}, {"port": 8083, "status": "stopped", "config": "..."}]`

    **Error Responses**:
    - `500 Internal Server Error`: If an unexpected error occurs during the batch processing logic.
    """
    try:
        return await get_batch_server_status_endpoint(request=request)
    except Exception as e:
        logger.error("Error in /batch-server-status endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching batch server status: {str(e)}")

@app.get("/benchmark", summary="Run Benchmark on a BitNet Model", tags=["Utilities"])
async def benchmark(
    model: ModelEnum = Query(..., description="The BitNet model to benchmark. See `ModelEnum` for available models."),
    n_token: int = Query(128, gt=0, description="Number of tokens to process in the benchmark. Must be > 0."),
    threads: int = Query(2, gt=0, description="Number of threads to use for the benchmark. Must be > 0."), # le=os.cpu_count() removed as it's handled by llama-bench
    n_prompt: int = Query(32, gt=0, description="Number of prompt tokens to use. Must be > 0.")
):
    """
    Runs a performance benchmark on a specified BitNet model.

    This endpoint executes the `llama-bench` utility with the given parameters.
    It's useful for evaluating the token generation speed and other performance
    metrics of a model.

    **Parameters**:
    - `model`: The model identifier (e.g., "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf").
    - `n_token`: Number of tokens to generate/process during the benchmark.
    - `threads`: Number of CPU threads for the benchmark tool.
    - `n_prompt`: Number of tokens in the initial prompt.

    **Successful Response (200 OK)**:
    - Parsed benchmark data, typically a list of dictionaries containing performance metrics.
      The exact structure depends on the output of `llama-bench`.

    **Error Responses**:
    - `500 Internal Server Error`: If the benchmark binary is not found or the benchmark process fails.
    """
    try:
        return await run_benchmark(model, n_token, threads, n_prompt)
    except Exception as e:
        logger.error(f"Error in /benchmark endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/perplexity", summary="Calculate Perplexity for a BitNet Model", tags=["Utilities"])
async def perplexity(
    model: ModelEnum = Query(..., description="The BitNet model for perplexity calculation. See `ModelEnum`."),
    prompt: str = Query(..., description="Input text (prompt) for perplexity calculation. Must meet minimum length requirements based on context size."),
    threads: int = Query(2, gt=0, description="Number of threads to use. Must be > 0."), # le=os.cpu_count() removed
    ctx_size: int = Query(10, gt=3, description="Context size for perplexity calculation. Must be > 3."),
    ppl_stride: int = Query(0, ge=0, description="Stride for perplexity calculation. 0 for full context.")
):
    """
    Calculates the perplexity of a given text using a specified BitNet model.

    This endpoint uses the `llama-perplexity` utility. Perplexity is a measure of how well a
    probability model predicts a sample. Lower perplexity usually indicates a better model fit.
    The input prompt must be sufficiently long relative to the context size.

    **Parameters**:
    - `model`: The model identifier.
    - `prompt`: The text for which to calculate perplexity.
    - `threads`: Number of CPU threads for the perplexity tool.
    - `ctx_size`: Context window size used in perplexity calculation.
    - `ppl_stride`: Stride used in perplexity calculation.

    **Successful Response (200 OK)**:
    - Parsed perplexity data, typically a JSON object containing metrics like PPL value.
      The exact structure depends on the output of `llama-perplexity`.

    **Error Responses**:
    - `400 Bad Request`: If the prompt is too short for the given context size.
    - `500 Internal Server Error`: If the perplexity binary is not found or the calculation process fails.
    """
    try:
        return await run_perplexity(model, prompt, threads, ctx_size, ppl_stride)
    except Exception as e:
        logger.error(f"Error in /perplexity endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-sizes", summary="Get Sizes of Available BitNet Models", tags=["Utilities"])
def model_sizes():
    """
    Retrieves the file sizes of all available .gguf model files in the 'models' directory.

    This endpoint scans the predefined models directory, identifies .gguf files,
    and returns their sizes in bytes, megabytes (MB), and gigabytes (GB).

    **Parameters**: None

    **Successful Response (200 OK)**:
    - A JSON object where keys are model filenames and values are objects
      containing `bytes`, `MB`, and `GB` sizes.
      Example: `{"ggml-model-i2_s.gguf": {"bytes": 2000000000, "MB": 1907.349, "GB": 1.863}}`
    """
    return get_model_sizes()

@app.post("/chat", summary="Chat with a BitNet Server Instance", tags=["Interaction"])
async def chat(chat_request: ChatRequest): # Renamed 'chat' to 'chat_request' to avoid conflict
    """
    Forwards a chat message to a specified, running BitNet server instance and returns its response.

    This acts as a middleman to a `llama-server` instance. The target server must be
    initialized and running on the specified port.

    **Request Body**:
    - `ChatRequest`: A JSON object containing:
        - `message` (str): The user's message/prompt.
        - `port` (int, default 8081): The port of the target BitNet server.
        - `threads` (int, default 1): Threads for this specific completion request (if supported by server).
        - `ctx_size` (int, default 2048): Context size for this request (if supported).
        - `n_predict` (int, default 256): Max tokens to predict for this request.
        - `temperature` (float, default 0.8): Temperature for this request.

    **Successful Response (200 OK)**:
    - The JSON response from the BitNet server, typically containing the generated text.
      The exact structure depends on the `llama-server`'s `/completion` endpoint.
      Example: `{"content": "This is the model's response." ...}`

    **Error Responses**:
    - `404 Not Found`: If the target BitNet server on the specified port is not running or not configured.
    - `503 Service Unavailable`: If the orchestrator cannot connect to the BitNet server.
    - `504 Gateway Timeout`: If the request to the BitNet server times out.
    - Other status codes from the BitNet server itself will be forwarded.
    - `500 Internal Server Error`: For other unexpected errors during the forwarding process.
    """
    try:
        return await chat_with_bitnet(chat_request)
    except HTTPException as e: # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        logger.error(f"Error in /chat endpoint for port {chat_request.port}: {str(e)}", exc_info=True)
        # Ensure a generic message for unexpected errors to avoid leaking details
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the chat request.")

# Parallel multi-chat endpoint
@app.post("/multichat", summary="Send Multiple Chat Requests to BitNet Servers", tags=["Interaction"])
async def multichat(multichat_request: MultiChatRequest):
    """
    Sends multiple chat messages to BitNet server instances concurrently.

    Each chat request in the batch can be targeted at a different (or the same) server port
    and can have its own parameters. This endpoint uses `asyncio.gather` to process
    the requests in parallel.

    **Request Body**:
    - `MultiChatRequest`: A JSON object containing a list of `requests`. Each item in this list
      is a `ChatRequest` object (see `/chat` endpoint for `ChatRequest` details).

    **Successful Response (200 OK)**:
    - A JSON object containing a `results` list. Each item in this list corresponds to a
      chat request from the input, containing either the successful JSON response from the
      BitNet server or an error object if that specific chat request failed.
      Example: `{"results": [{"content": "..."}, {"error": "Server on port 8083 not running...", "status_code": 404}]}`

    **Error Responses**:
    - `500 Internal Server Error`: If an unexpected error occurs during the overall batch processing.
      Individual chat failures are reported within the `results` list in the 200 OK response.
    """
    try:
        return await multichat_with_bitnet(multichat_request)
    except Exception as e:
        logger.error(f"Error in /multichat endpoint: {str(e)}", exc_info=True)
        # Ensure a generic message for unexpected errors
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the multichat request.")

# Wrap with MCP for Model Context Protocol support
mcp = FastApiMCP(app)

# Mount the MCP server directly to your FastAPI app
mcp.mount()