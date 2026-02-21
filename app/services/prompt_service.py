from typing import Optional, Dict
from app.configs.supabase import supabase_client

class PromptService:
    def __init__(self):
        self.client = supabase_client

    async def get_latest_prompt(self, prompt_name: str = None) -> Optional[Dict]:
        """
        Fetches the latest active prompt template from 'prompt_template' table.
        If prompt_name is None, fetches the latest active prompt regardless of name.
        """
        if not self.client:
            return None

        try:
            # Build query
            query = self.client.table("prompt_template").select("*")
            
            # Filter by name if provided
            if prompt_name:
                query = query.eq("name", prompt_name)
            
            # Fetch latest active prompt (by version DESC, then created_at DESC)
            response = query\
                .eq("is_active", True)\
                .order("version", desc=True)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
        except Exception as e:
            print(f"Error fetching prompt: {e}")
            return None

prompt_service = PromptService()
