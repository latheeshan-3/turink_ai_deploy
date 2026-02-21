from app.configs.redis import redis_client
from typing import List, Dict, Optional
import json

class RedisService:
    def __init__(self):
        self.client = redis_client
        self.expiration = 3600  # 1 hour default
        self.conversation_expiration = 86400  # 24 hours for conversations

    async def get_cache(self, key: str):
        try:
            return await self.client.get(key)
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    async def set_cache(self, key: str, value: str, expire: int = None):
        try:
            await self.client.set(
                key, 
                value, 
                ex=expire if expire else self.expiration
            )
        except Exception as e:
            print(f"Redis set error: {e}")

    # ========== Conversation Memory Methods ==========
    
    async def store_conversation_message(self, conversation_id: str, role: str, content: str):
        """
        Append a message to the conversation history in Redis.
        
        Args:
            conversation_id: Unique conversation identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        try:
            key = f"conversation:{conversation_id}:messages"
            message = {"role": role, "content": content}
            
            # Push message to Redis list
            await self.client.rpush(key, json.dumps(message))
            
            # Set expiration on the key
            await self.client.expire(key, self.conversation_expiration)
            
            print(f"Stored {role} message for conversation {conversation_id}")
        except Exception as e:
            print(f"Error storing conversation message: {e}")

    async def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """
        Retrieve full conversation history from Redis.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        try:
            key = f"conversation:{conversation_id}:messages"
            messages = await self.client.lrange(key, 0, -1)
            
            if not messages:
                return []
            
            # Parse JSON messages
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

    async def get_message_count(self, conversation_id: str) -> int:
        """
        Count the number of user messages in the conversation.
        
        Returns:
            Number of user messages
        """
        try:
            messages = await self.get_conversation_history(conversation_id)
            user_message_count = sum(1 for msg in messages if msg.get("role") == "user")
            return user_message_count
        except Exception as e:
            print(f"Error counting messages: {e}")
            return 0

    async def store_conversation_summary(self, conversation_id: str, summary: str):
        """
        Store a summarized version of the conversation.
        This replaces the messages list with a summary after 5+ messages.
        
        Args:
            conversation_id: Unique conversation identifier
            summary: LLM-generated summary of the conversation
        """
        try:
            summary_key = f"conversation:{conversation_id}:summary"
            await self.client.set(summary_key, summary, ex=self.conversation_expiration)
            
            # Clear old messages after summarization
            messages_key = f"conversation:{conversation_id}:messages"
            await self.client.delete(messages_key)
            
            print(f"Stored conversation summary for {conversation_id}")
        except Exception as e:
            print(f"Error storing conversation summary: {e}")

    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        Retrieve the conversation summary if it exists.
        
        Returns:
            Summary string or None
        """
        try:
            summary_key = f"conversation:{conversation_id}:summary"
            summary = await self.client.get(summary_key)
            return summary
        except Exception as e:
            print(f"Error retrieving conversation summary: {e}")
            return None

    async def clear_conversation(self, conversation_id: str):
        """
        Clear all conversation data (messages and summary).
        """
        try:
            messages_key = f"conversation:{conversation_id}:messages"
            summary_key = f"conversation:{conversation_id}:summary"
            
            await self.client.delete(messages_key)
            await self.client.delete(summary_key)
            
            print(f"Cleared conversation data for {conversation_id}")
        except Exception as e:
            print(f"Error clearing conversation: {e}")

redis_service = RedisService()

