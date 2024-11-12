from fastapi import FastAPI, HTTPException, Query, Depends
from .models import ModelEnum, BenchmarkRequest, PerplexityRequest, InferenceRequest
from .utils import run_command, parse_benchmark_data, parse_perplexity_data
import os
import subprocess

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

def validate_prompt_length(prompt: str = Query(..., description="Input text for perplexity calculation"), ctx_size: int = Query(10, gt=0)) -> str:
    token_count = len(prompt.split())
    min_tokens = 2 * ctx_size
    
    if token_count < min_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too short. Needs at least {min_tokens} tokens, got {token_count}"
        )
    return prompt

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

async def run_inference_endpoint(
    model: ModelEnum,
    n_predict: int = Query(128, gt=0, le=100000),
    prompt: str = "",
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(2048, gt=0),
    temperature: float = Query(0.8, gt=0.0, le=2.0)
):
    """Endpoint to run inference with the given parameters."""
    request = InferenceRequest(
        model=model,
        n_predict=n_predict,
        prompt=prompt,
        threads=threads,
        ctx_size=ctx_size,
        temperature=temperature
    )
    output = run_inference(request)
    return {"result": output}

def run_inference(args: InferenceRequest) -> str:
    """Run the inference command with the given arguments."""
    build_dir = os.getenv("BUILD_DIR", "build")
    main_path = os.path.join(build_dir, "bin", "llama-cli")

    if not os.path.exists(main_path):
        raise HTTPException(status_code=500, detail="Inference binary not found")
    
    command = [
        main_path,
        '-m', args.model.value,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        "-b", "1"
    ]
    output = run_command(command)
    return output