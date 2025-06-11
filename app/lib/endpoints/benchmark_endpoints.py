import os
import asyncio
import logging
from ..models import ModelEnum, BenchmarkRequest, PerplexityRequest
from ..utils import parse_benchmark_data, parse_perplexity_data

import os
import subprocess # Keep for CalledProcessError
import asyncio # Ensure asyncio is imported
from pydantic import BaseModel, Field
from fastapi import HTTPException, Query, Depends
import logging # Import logging

# --- Logging Configuration for this module ---
logger = logging.getLogger(__name__)

def validate_prompt_length(prompt: str = Query(..., description="Input text for perplexity calculation"), ctx_size: int = Query(10, gt=3)) -> str:
    token_count = len(prompt.split())
    min_tokens = 2 * ctx_size
    if token_count < min_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too short. Needs at least {min_tokens} tokens, got {token_count}"
        )
    return prompt

async def run_benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0),
    n_prompt: int = Query(32, gt=0)
):
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
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Benchmark failed: {stderr_bytes.decode(errors='ignore')}")
        parsed_data = parse_benchmark_data(stdout_bytes.decode(errors='ignore'))
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during benchmark: {str(e)}")

async def run_perplexity(
    model: ModelEnum,
    prompt: str = Depends(validate_prompt_length),
    threads: int = Query(2, gt=0),
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

async def run_benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0),
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
