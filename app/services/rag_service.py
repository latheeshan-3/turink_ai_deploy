import os
import time
from typing import List, Optional, Any
from datetime import timedelta
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from google.cloud import aiplatform
from vertexai.preview import caching
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self):
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project, location=self.location)
        
        self.chat_model_name = os.getenv("VERTEX_CHAT_MODEL", "gemini-1.5-flash")
        self.embed_model_name = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")
        
        self.embeddings_client = VertexAIEmbeddings(model_name=self.embed_model_name)
        # Default LLM client (without cache)
        self.llm_client = ChatVertexAI(model_name=self.chat_model_name)

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            return await self.embeddings_client.aembed_query(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    async def generate_response(
        self, 
        user_query: str, 
        context_chunks: List[str], 
        system_instruction: str,
        cached_content_name: Optional[str] = None
    ) -> str:
        """
        Generates response using Vertex AI. 
        If cached_content_name is provided, uses context caching.
        Otherwise, uses standard generation.
        """
        try:
            # Prepare context from chunks
            context_str = "\n\n".join(context_chunks)
            full_prompt = f"Context:\n{context_str}\n\nUser Question: {user_query}"

            if cached_content_name:
                # Cache has already been validated by chat_service
                # Use existing cache (system instruction is cached)
                print(f"Using Vertex AI Context Cache: {cached_content_name}")
                
                # Extract only the cache ID from the full resource path
                # ChatVertexAI will automatically construct the full path
                # Format: projects/{project}/locations/{location}/cachedContents/{cache_id}
                cache_id = cached_content_name.split('/')[-1] if '/' in cached_content_name else cached_content_name
                print(f"Extracted cache ID: {cache_id}")
                
                # Create model instance with cached content (pass only the ID)
                model_with_cache = ChatVertexAI(
                    model_name=self.chat_model_name,
                    cached_content=cache_id
                )
                
                response = await model_with_cache.ainvoke(full_prompt)
                return response.content
            else:
                # Standard generation (context in prompt with system instruction)
                messages = [
                    ("system", system_instruction),
                    ("human", full_prompt)
                ]
                response = await self.llm_client.ainvoke(messages)
                return response.content

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating the response."
            
    async def create_context_cache(
        self, 
        system_instruction: str, 
        ttl_hours: int = 1
    ) -> Optional[tuple[str, str]]:
        """
        Creates a Vertex AI context cache for the system instruction.
        Returns tuple of (cache_resource_name, expire_time) or None.
        
        NOTE: Vertex AI requires 'contents' parameter with at least one user content.
        We provide a placeholder to enable caching of the system instruction.
        """
        try:
            print(f"Creating Vertex AI context cache with TTL: {ttl_hours} hours")
            
            # Vertex AI Context Caching requires contents parameter with at least one user message
            # We provide a placeholder content to cache the system instruction
            from vertexai.generative_models import Content, Part
            
            placeholder_content = Content(
                role="user",
                parts=[Part.from_text("Context caching placeholder - this enables system instruction caching")]
            )
            
            # Create cached content with system instruction and placeholder
            cached_content = caching.CachedContent.create(
                model_name=self.chat_model_name,
                system_instruction=system_instruction,
                contents=[placeholder_content],  # Required: at least one user content
                ttl=timedelta(hours=ttl_hours),
            )
            
            # Get the full resource name
            # The resource name format is: projects/{project}/locations/{location}/cachedContents/{cache_id}
            cache_resource_name = cached_content.resource_name if hasattr(cached_content, 'resource_name') else cached_content.name
            
            # Get expiration time
            expire_time = cached_content.expire_time.isoformat() if hasattr(cached_content, 'expire_time') else None
            
            print(f"Cache created successfully!")
            print(f"  Cache ID: {cached_content.name}")
            print(f"  Resource name: {cache_resource_name}")
            print(f"  Expires at: {expire_time}")
            
            # Return both cache name and expire time
            return (cache_resource_name, expire_time)
            
        except Exception as e:
            print(f"Error creating context cache: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard generation without caching")
            return None

    async def validate_cache_exists(self, cache_name: str) -> bool:
        """
        Validates if a cache exists in Vertex AI by attempting to retrieve it.
        
        Args:
            cache_name: Full cache resource name or just the cache ID
            
        Returns:
            True if cache exists and is valid, False otherwise
        """
        try:
            # Extract cache ID if full resource name provided
            cache_id = cache_name.split('/')[-1] if '/' in cache_name else cache_name
            
            # Try to get the cached content
            cached_content = caching.CachedContent.get(cache_id)
            
            print(f"Cache validation successful: {cache_id}")
            return True
        except Exception as e:
            print(f"Cache validation failed for {cache_name}: {e}")
            return False


rag_service = RAGService()
