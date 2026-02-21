from typing import Optional
from pydantic import BaseModel, Field

class ChatResponse(BaseModel):
    message: str = Field(..., description="AI generated response")

class EmbeddingResponse(BaseModel):
    state: bool = Field(..., description="Success status of embedding generation")
    message: str = Field(..., description="Status message")

class PromptActivationResponse(BaseModel):
    success: bool = Field(..., description="Whether activation succeeded")
    message: str = Field(..., description="Status message")
    cache_name: Optional[str] = Field(None, description="Created cache name if applicable")
