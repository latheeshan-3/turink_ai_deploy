from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    message: str = Field(..., description="User's query message")

class EmbeddingRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID (UUID) to process for embeddings")

class PromptActivationRequest(BaseModel):
    prompt_id: str = Field(..., description="Prompt ID (UUID) to activate and cache")
