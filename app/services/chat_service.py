import json
from app.services.redis_service import redis_service
from app.services.vectorstore_service import vectorstore_service
from app.services.prompt_service import prompt_service
from app.services.cache_service import cache_service
from app.services.rag_service import rag_service

class ChatService:
    async def process_chat(self, conversation_id: str, user_query: str) -> dict:
        # 1. Store user message in conversation history
        await redis_service.store_conversation_message(conversation_id, "user", user_query)
        print(f"Stored user message for conversation {conversation_id}")
        
        # 2. Get conversation context (either summary or full history)
        conversation_context = await self._get_conversation_context(conversation_id)
        
        # 3. Check message count and summarize if needed (after 5 user messages)
        user_message_count = await redis_service.get_message_count(conversation_id)
        print(f"User message count: {user_message_count}")
        
        if user_message_count >= 5:
            print("Message count >= 5, triggering summarization...")
            await self._summarize_conversation(conversation_id)
            # After summarization, update conversation context
            conversation_context = await self._get_conversation_context(conversation_id)

        # 4. Check Redis Cache for exact query (optional response caching)
        cache_key = f"chat:{conversation_id}:{hash(user_query)}"
        cached_response = await redis_service.get_cache(cache_key)
        if cached_response:
             print("Redis Cache Hit")
             # Still store the cached response in conversation
             await redis_service.store_conversation_message(conversation_id, "assistant", cached_response)
             return {"message": cached_response, "source": "redis_cache"}

        print("Redis Cache Miss - Proceeding to Semantic Search")

        # 5. Embedding & Vector Search
        embedding = await rag_service.generate_embedding(user_query)
        relevant_chunks = await vectorstore_service.get_relevant_chunks(embedding)
        
        print(f"Vector Search: Retrieved {len(relevant_chunks)} chunks")
        
        chunk_texts = [chunk.get('content', '') for chunk in relevant_chunks]

        # 6. Fetch Latest Prompt Template (by version DESC)
        prompt_data = await prompt_service.get_latest_prompt()
        if not prompt_data:
            system_instruction = "You are a helpful AI assistant."
            prompt_id = None
            print("No prompt template found, using default instruction")
        else:
            system_instruction = prompt_data.get("template_content", "You are a helpful AI assistant.")
            prompt_id = str(prompt_data.get("id"))
            print(f"Using prompt template: name='{prompt_data.get('name')}', version={prompt_data.get('version')}, id={prompt_id}")
            print(f"System instruction preview: {system_instruction[:100]}...")

        # 7. Check & Validate Vertex AI Context Cache (with Vertex AI validation)
        cache_name = None
        if prompt_id:
            cache_name = await cache_service.validate_and_get_cache(prompt_id, rag_service)
        
        if not cache_name and prompt_id:
            print("GCP Cache Miss, Expired, or Invalid - Creating new Context Cache")
            cache_result = await rag_service.create_context_cache(system_instruction)
            
            if cache_result:
                new_cache_name, expire_time = cache_result
                await cache_service.save_cached_context(prompt_id, new_cache_name, expire_time)
                cache_name = new_cache_name
                print(f"New cache created and saved: {cache_name}")
        elif cache_name:
            print(f"GCP Cache Valid: {cache_name}")

        # 8. Prepare context with conversation history
        # Combine conversation context with retrieved chunks
        context_with_conversation = self._build_context_with_conversation(
            conversation_context, 
            chunk_texts
        )

        # 9. Generate Response (RAG with conversation context)
        ai_response = await rag_service.generate_response(
            user_query=user_query,
            context_chunks=context_with_conversation,
            system_instruction=system_instruction,
            cached_content_name=cache_name
        )

        # 10. Store AI response in conversation history
        await redis_service.store_conversation_message(conversation_id, "assistant", ai_response)

        # 11. Store in Redis response cache
        await redis_service.set_cache(cache_key, ai_response)

        return {"message": ai_response, "source": "generated"}

    async def _get_conversation_context(self, conversation_id: str) -> str:
        """
        Get conversation context - either summary or full message history.
        """
        # First check if there's a summary
        summary = await redis_service.get_conversation_summary(conversation_id)
        if summary:
            print(f"Using conversation summary for context")
            return f"Previous conversation summary: {summary}"
        
        # Otherwise get full message history
        messages = await redis_service.get_conversation_history(conversation_id)
        if not messages:
            return ""
        
        # Format messages as conversation context
        conversation_text = "Previous conversation:\n"
        for msg in messages[:-1]:  # Exclude the current user message (last one)
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"{role.capitalize()}: {content}\n"
        
        print(f"Using full conversation history ({len(messages)-1} messages)")
        return conversation_text

    async def _summarize_conversation(self, conversation_id: str):
        """
        Summarize the conversation using LLM after 5+ messages.
        """
        messages = await redis_service.get_conversation_history(conversation_id)
        if not messages:
            return
        
        # Build conversation text
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"{role.capitalize()}: {content}\n\n"
        
        # Call LLM to summarize
        summarization_prompt = f"""Summarize the following conversation concisely, preserving all important context and details that would be needed to continue the conversation naturally:

{conversation_text}

Provide a brief summary (2-3 sentences) that captures the key points and current state of the conversation."""
        
        try:
            summary = await rag_service.llm_client.ainvoke([("human", summarization_prompt)])
            summary_text = summary.content
            
            # Store summary and clear messages
            await redis_service.store_conversation_summary(conversation_id, summary_text)
            print(f"Conversation summarized: {summary_text[:100]}...")
        except Exception as e:
            print(f"Error summarizing conversation: {e}")

    def _build_context_with_conversation(self, conversation_context: str, chunk_texts: list) -> list:
        """
        Build context list that includes conversation history and retrieved chunks.
        """
        context_items = []
        
        # Add conversation context first if it exists
        if conversation_context:
            context_items.append(conversation_context)
        
        # Add retrieved chunks
        context_items.extend(chunk_texts)
        
        return context_items

chat_service = ChatService()

