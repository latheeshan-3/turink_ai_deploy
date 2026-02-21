from typing import Dict, Optional
from app.configs.supabase import supabase_client
from app.services.rag_service import rag_service
from app.services.cache_service import cache_service


class PromptUpdateService:
    def __init__(self):
        self.client = supabase_client

    async def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict]:
        """
        Fetches a prompt template by its ID.
        """
        if not self.client:
            return None

        try:
            response = self.client.table("prompt_template")\
                .select("*")\
                .eq("id", prompt_id)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
        except Exception as e:
            print(f"Error fetching prompt by ID: {e}")
            return None

    async def activate_prompt(self, prompt_id: str) -> Dict:
        """
        Activates a prompt and creates a Vertex AI cache for it.
        
        1. Fetch the prompt template by ID
        2. Create Vertex AI context cache with the template content
        3. Save cache reference to gcp_cache table
        
        Args:
            prompt_id: UUID of the prompt to activate
            
        Returns:
            Dict with success status, message, and cache_name
        """
        try:
            # Step 1: Fetch the prompt template
            prompt = await self.get_prompt_by_id(prompt_id)
            
            if not prompt:
                return {
                    "success": False,
                    "message": f"Prompt with ID {prompt_id} not found",
                    "cache_name": None
                }
            
            template_content = prompt.get("template_content", "")
            prompt_name = prompt.get("name", "Unknown")
            
            if not template_content:
                return {
                    "success": False,
                    "message": "Prompt template content is empty",
                    "cache_name": None
                }
            
            print(f"Activating prompt: {prompt_name} (ID: {prompt_id})")
            
            # Step 2: Create Vertex AI context cache
            cache_result = await rag_service.create_context_cache(
                system_instruction=template_content,
                ttl_hours=1  # Default TTL of 1 hour
            )
            
            if not cache_result:
                return {
                    "success": False,
                    "message": "Failed to create Vertex AI context cache",
                    "cache_name": None
                }
            
            cache_name, expire_time = cache_result
            
            # Step 3: Save cache to gcp_cache table
            await cache_service.save_cached_context(
                prompt_id=prompt_id,
                cache_name=cache_name,
                expire_time=expire_time
            )
            
            print(f"Successfully activated prompt and created cache: {cache_name}")
            
            return {
                "success": True,
                "message": f"Prompt '{prompt_name}' activated and cached successfully",
                "cache_name": cache_name
            }
            
        except Exception as e:
            print(f"Error activating prompt: {e}")
            return {
                "success": False,
                "message": f"Error activating prompt: {str(e)}",
                "cache_name": None
            }


prompt_update_service = PromptUpdateService()
