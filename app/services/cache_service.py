from typing import Optional
from datetime import datetime, timezone
from app.configs.supabase import supabase_client

class CacheService:
    def __init__(self):
        self.client = supabase_client


    async def validate_and_get_cache(self, prompt_id: str, rag_service=None) -> Optional[str]:
        """
        Checks 'gcp_cache' table for an active cache and validates:
        1. Expiration time in database
        2. Actual existence in Vertex AI (if rag_service provided)
        
        Returns cache_name only if valid and exists in Vertex AI.
        """
        if not self.client or prompt_id is None:
            return None

        try:
            response = self.client.table("gcp_cache")\
                .select("cache_name, expire_time")\
                .eq("prompt_id", prompt_id)\
                .eq("is_active", True)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                cache_data = response.data[0]
                cache_name = cache_data["cache_name"]
                expire_time = cache_data.get("expire_time")
                
                # Check if cache has expired in database
                if expire_time:
                    # Parse expire_time (it comes as ISO string from Supabase)
                    expire_dt = datetime.fromisoformat(expire_time.replace('Z', '+00:00'))
                    current_dt = datetime.now(timezone.utc)
                    
                    if current_dt >= expire_dt:
                        print(f"Cache expired in database: {cache_name} (expired at {expire_time})")
                        await self.invalidate_cache(cache_name)
                        return None
                
                # Also validate cache exists in Vertex AI
                if rag_service:
                    cache_exists = await rag_service.validate_cache_exists(cache_name)
                    if not cache_exists:
                        print(f"Cache not found in Vertex AI (404): {cache_name}")
                        await self.invalidate_cache(cache_name)
                        return None
                
                return cache_name
            
            return None
        except Exception as e:
            print(f"Error validating GCP cache: {e}")
            return None


    async def invalidate_cache(self, cache_name: str):
        """
        Marks a cache as inactive in the database.
        """
        if not self.client:
            return

        try:
            self.client.table("gcp_cache")\
                .update({"is_active": False})\
                .eq("cache_name", cache_name)\
                .execute()
            print(f"Marked cache as inactive: {cache_name}")
        except Exception as e:
            print(f"Error invalidating cache: {e}")

    async def get_cached_context_name(self, prompt_id: str) -> Optional[str]:
        """
        DEPRECATED: Use validate_and_get_cache instead.
        Checks 'gcp_cache' table if there is an active Vertex AI cache for this prompt_id.
        """
        if not self.client:
            return None

        if prompt_id is None:
            return None

        try:
            response = self.client.table("gcp_cache")\
                .select("cache_name")\
                .eq("prompt_id", prompt_id)\
                .eq("is_active", True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]["cache_name"]
            
            return None
        except Exception as e:
            print(f"Error checking GCP cache: {e}")
            return None

    async def save_cached_context(self, prompt_id: str, cache_name: str, expire_time: str):
        """
        Saves the new cache_name and expire_time to 'gcp_cache' table.
        Deactivates all existing active caches first to ensure only one cache is active.
        """
        if not self.client:
            return

        if prompt_id is None:
            return

        try:
            # First, deactivate all existing active caches
            self.client.table("gcp_cache")\
                .update({"is_active": False})\
                .eq("is_active", True)\
                .execute()
            print("Deactivated all existing active caches")

            # Now insert the new cache as active
            data = {
                "prompt_id": prompt_id,
                "cache_name": cache_name,
                "is_active": True,
                "expire_time": expire_time  # Always store expire time
            }

            result = self.client.table("gcp_cache").insert(data).execute()
            print(f"Successfully saved cache to database: prompt_id={prompt_id}, cache_name={cache_name}, expires={expire_time}")
        except Exception as e:
            print(f"Error saving GCP cache record: {e}")

cache_service = CacheService()
