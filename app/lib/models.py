from typing import Dict, Any
from pydantic import BaseModel, validator, root_validator
from enum import Enum
import os

def create_model_enum(directory: str):
    """Dynamically create an Enum for models based on files in the directory."""
    models = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".gguf"):
                    model_name = f"{subdir}_{file.replace('-', '_').replace('.', '_')}"
                    models[model_name] = os.path.join(subdir_path, file)
    return Enum("ModelEnum", models)

# Create the ModelEnum based on the models directory
ModelEnum = create_model_enum("models")

max_n_predict = 100000

class BenchmarkRequest(BaseModel):
    model: ModelEnum
    n_token: int = 128
    threads: int = 2
    n_prompt: int = 32
    
    @validator('threads')
    def validate_threads(cls, v):
        max_threads = os.cpu_count()
        if v > max_threads:
            raise ValueError(f"Number of threads cannot exceed {max_threads}")
        return v
    
    @validator('n_token', 'n_prompt', 'threads')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

class PerplexityRequest(BaseModel):
    model: ModelEnum
    prompt: str
    threads: int = 2
    ctx_size: int = 3
    ppl_stride: int = 0

    @validator('threads')
    def validate_threads(cls, v):
        max_threads = os.cpu_count()
        if v > max_threads:
            raise ValueError(f"Number of threads cannot exceed {max_threads}")
        elif v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @validator('ctx_size')
    def validate_positive(cls, v):
        if v < 3:
            raise ValueError("Value must be greater than 3")
        return v

    @root_validator(pre=True)
    def validate_prompt_length(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        prompt = values.get('prompt')
        ctx_size = values.get('ctx_size')
        
        if prompt and ctx_size:
            token_count = len(prompt.split())
            min_tokens = 2 * ctx_size
            
            if token_count < min_tokens:
                raise ValueError(f"Prompt too short. Needs at least {min_tokens} tokens, got {token_count}")
        
        return values

class InferenceRequest(BaseModel):
    model: ModelEnum
    n_predict: int = 128
    prompt: str
    threads: int = 2
    ctx_size: int = 2048
    temperature: float = 0.8

    @validator('threads')
    def validate_threads(cls, v):
        max_threads = os.cpu_count()
        if v > max_threads:
            raise ValueError(f"Number of threads cannot exceed {max_threads}")
        return v

    @validator('n_predict')
    def validate_n_predict(cls, v):
        if v > max_n_predict:
            raise ValueError(f"Number of predictions cannot exceed {max_n_predict}")
        return v
    
    @validator('threads', 'ctx_size', 'temperature', 'n_predict')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v