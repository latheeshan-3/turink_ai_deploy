from typing import List, Dict, Any
from app.configs.supabase import supabase_client

class VectorStoreService:
    def __init__(self):
        self.client = supabase_client

    async def get_relevant_chunks(self, embedding: List[float], match_threshold: float = 0.5, match_count: int = 5) -> List[Dict[str, Any]]:
        """
        Calls the Supabase RPC function 'match_documents' to find relevant chunks.
        """
        if not self.client:
            print("Supabase client not initialized.")
            return []

        try:
            # RPC call to match_documents
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                }
            ).execute()
            
            return response.data if response.data else []
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []

vectorstore_service = VectorStoreService()
