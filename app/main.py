from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.models.request import ChatRequest, EmbeddingRequest, PromptActivationRequest
from app.models.response import ChatResponse, EmbeddingResponse, PromptActivationResponse
from app.services.chat_service import chat_service
from app.services.ingestion_pipeline import ingestion_pipeline
from app.services.prompt_update_service import prompt_update_service

app = FastAPI(title="Turing Labs Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = await chat_service.process_chat(request.conversation_id, request.message)
        return ChatResponse(message=result["message"])
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings_endpoint(request: EmbeddingRequest):
    """
    Generate embeddings for a document and store in vector database.
    
    Args:
        request: EmbeddingRequest with doc_id
        
    Returns:
        EmbeddingResponse with success state and message
    """
    try:
        print(f"Processing embedding request for document: {request.doc_id}")
        result = ingestion_pipeline(request.doc_id)
        
        if result["success"]:
            return EmbeddingResponse(
                state=True, 
                message=f"Embeddings generated successfully. Processed {result['chunks_processed']} chunks."
            )
        else:
            return EmbeddingResponse(
                state=False, 
                message=f"Embedding generation failed: {result['error']}"
            )
    except Exception as e:
        print(f"Error in embeddings endpoint: {e}")
        return EmbeddingResponse(
            state=False,
            message=f"Internal error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/prompt", response_model=PromptActivationResponse)
async def prompt_activation_endpoint(request: PromptActivationRequest):
    """
    Activate a prompt and create Vertex AI context cache.
    
    Args:
        request: PromptActivationRequest with prompt_id
        
    Returns:
        PromptActivationResponse with success state, message, and cache_name
    """
    try:
        print(f"Processing prompt activation request for: {request.prompt_id}")
        result = await prompt_update_service.activate_prompt(request.prompt_id)
        
        return PromptActivationResponse(
            success=result["success"],
            message=result["message"],
            cache_name=result.get("cache_name")
        )
    except Exception as e:
        print(f"Error in prompt activation endpoint: {e}")
        return PromptActivationResponse(
            success=False,
            message=f"Internal error: {str(e)}",
            cache_name=None
        )
