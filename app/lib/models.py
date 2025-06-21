from typing import Dict, Any, Optional
from pydantic import BaseModel, validator, root_validator, Field
from enum import Enum
import os
from .utils import is_safe_cli_alias

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

# --- Llama CLI Models ---

class LlamaCliInitRequest(BaseModel):
    cli_alias: str = Field(..., description="A unique alias for this llama-cli configuration.", example="bitnet_001")
    threads: int = Field(default=1, gt=0, description="Number of threads to use for llama-cli. Defaults to half of CPU cores.", example=4)
    ctx_size: int = Field(default=2048, gt=0, description="Context size (in tokens) for llama-cli.", example=2048)
    n_predict: int = Field(default=256, gt=-2, description="Max tokens to predict with llama-cli. -1 for infinite, -2 for default.", example=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Temperature for llama-cli sampling.", example=0.7)
    top_k: Optional[int] = Field(default=None, gt=0, description="Top-k sampling parameter.")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter.")
    repeat_penalty: Optional[float] = Field(default=None, gt=0.0, description="Repeat penalty.")
    system_prompt: Optional[str] = Field(default="You are a helpful assistant.", description="An optional system prompt to guide the model's behavior.")

    @validator('cli_alias')
    def validate_cli_alias(cls, v):
        if not is_safe_cli_alias(v):
            raise ValueError('cli_alias contains invalid characters. Use only alphanumeric, hyphens, and underscores.')
        return v

class LlamaCliInstanceInfo(BaseModel):
    cli_alias: str
    status: str # e.g., "active", "inactive"
    config: LlamaCliInitRequest

class LlamaCliChatRequest(BaseModel):
    cli_alias: str = Field(..., description="The alias of the initialized llama-cli configuration to use.", example="bitnet_001")
    prompt: str = Field(..., description="The user's prompt/message to send to llama-cli.", example="What is the capital of France?")
    n_predict: Optional[int] = Field(default=None, gt=-2, description="Override max tokens to predict for this specific request. -1 for infinite.")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Override temperature for this specific request.")
    top_k: Optional[int] = Field(default=None, gt=0, description="Override top-k for this request.")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override top-p for this request.")
    repeat_penalty: Optional[float] = Field(default=None, gt=0.0, description="Override repeat penalty for this request.")

    @validator('cli_alias')
    def validate_cli_alias(cls, v):
        if not is_safe_cli_alias(v):
            raise ValueError('cli_alias contains invalid characters. Use only alphanumeric, hyphens, and underscores.')
        return v

class BatchLlamaCliInitRequest(BaseModel):
    requests: list[LlamaCliInitRequest] = Field(..., description="A list of llama-cli configurations to initialize.")

class BatchLlamaCliRemoveRequest(BaseModel):
    aliases: list[str] = Field(..., description="A list of llama-cli configuration aliases to remove.")

    @validator('aliases', each_item=True)
    def validate_aliases(cls, v):
        if not is_safe_cli_alias(v):
            raise ValueError('An alias in the list contains invalid characters. Use only alphanumeric, hyphens, and underscores.')
        return v

class BatchLlamaCliStatusRequest(BaseModel):
    aliases: list[str] = Field(..., description="A list of llama-cli configuration aliases to check the status of.")

    @validator('aliases', each_item=True)
    def validate_aliases(cls, v):
        if not is_safe_cli_alias(v):
            raise ValueError('An alias in the list contains invalid characters. Use only alphanumeric, hyphens, and underscores.')
        return v

class BatchLlamaCliChatRequest(BaseModel):
    requests: list[LlamaCliChatRequest] = Field(..., description="A list of chat requests to process.")