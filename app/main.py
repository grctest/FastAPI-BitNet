from fastapi import FastAPI, Query
import os

from lib.models import ModelEnum
from lib.endpoints import run_benchmark, run_perplexity, get_model_sizes, run_inference_endpoint

app = FastAPI()

@app.get("/benchmark")
async def benchmark(
    model: ModelEnum,
    n_token: int = Query(128, gt=0),
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    n_prompt: int = Query(32, gt=0)
):
    return await run_benchmark(model, n_token, threads, n_prompt)

@app.get("/perplexity")
async def perplexity(
    model: ModelEnum,
    prompt: str,
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(4, gt=0),
    ppl_stride: int = Query(0, ge=0)
):
    return await run_perplexity(model, prompt, threads, ctx_size, ppl_stride)

@app.get("/run-inference")
async def inference(
    model: ModelEnum,
    n_predict: int = Query(128, gt=0, le=100000),
    prompt: str = "",
    threads: int = Query(2, gt=0, le=os.cpu_count()),
    ctx_size: int = Query(2048, gt=0),
    temperature: float = Query(0.8, gt=0.0, le=2.0)
):
    return await run_inference_endpoint(model, n_predict, prompt, threads, ctx_size, temperature)

@app.get("/model-sizes")
def model_sizes():
    return get_model_sizes()