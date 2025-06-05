from .endpoints import ChatRequest
from typing import List
from pydantic import BaseModel

__all__ = ["ChatRequest", "MultiChatRequest"]

# Re-export for import convenience
class MultiChatRequest(BaseModel):
    requests: List[ChatRequest]
