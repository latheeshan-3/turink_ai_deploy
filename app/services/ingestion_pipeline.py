import os
from dotenv import load_dotenv
from app.configs.supabase import supabase_client
from app.services.supabase_service import (
    document_data_fetcher,
    supabase_storage_loader
)
from app.services.embedding_service import EmbeddingService
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def ingestion_pipeline(doc_id: str):
    """
    Main ingestion pipeline that processes a document for embedding generation.
    
    Workflow:
    1. Fetch document metadata and URL from Supabase database
    2. Load document content from Supabase Storage
    3. Chunk the document into smaller pieces
    4. Add metadata to each chunk
    5. Generate embeddings and store in chunk_documents table
    
    Args:
        doc_id: Document UUID as string
        
    Returns:
        Dict with keys:
            - success: bool indicating success/failure
            - chunks_processed: int (if successful)
            - error: str (if failed)
    """
    try:
        print(f"Fetching document data with id: {doc_id}")
        doc_metadata, doc_access_url = document_data_fetcher.fetch_document_data_by_id(
            doc_id
        )

        print(f"Loading document from: {doc_access_url}")
        document = supabase_storage_loader.load_from_supabase_url(doc_access_url)

        print("Chunking document")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(document)
        print(f"Chunks: {len(chunks)}")

        print("Adding metadata to each chunk")
        for i, chunk in enumerate(chunks):
            original_content = chunk.page_content
            custom_text = f'[This content is from the {doc_metadata.get("title", "")}] - {original_content}'
            chunk.page_content = custom_text

            chunk.metadata.update(
                {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "title": doc_metadata.get("title", ""),
                    "source_type": doc_metadata.get("source_type", ""),
                    "source_path": doc_metadata.get("source_path", ""),
                }
            )
        print("Metadata added to each chunk")

        # Initialize embedding service based on LLM_PROVIDER env var
        embedding_service = EmbeddingService()
        embeddings = embedding_service.get_embeddings()

        print("Generating embeddings for chunks")
        # Batch generate embeddings for all chunks at once (more efficient)
        chunk_texts = [chunk.page_content for chunk in chunks]
        embedding_vectors = embeddings.embed_documents(chunk_texts)
        
        # Prepare rows for manual insertion to ensure document_id is populated
        rows_to_insert = []
        for i, chunk in enumerate(chunks):
            # Prepare row with explicit document_id field
            row = {
                "content": chunk.page_content,
                "embedding": embedding_vectors[i],
                "metadata": chunk.metadata,
                "document_id": doc_id  # Explicitly set document_id from function parameter
            }
            rows_to_insert.append(row)

        print(f"Inserting {len(rows_to_insert)} chunks into chunk_documents table")
        # Insert all chunks in batch
        result = supabase_client.table("chunk_documents").insert(rows_to_insert).execute()
        
        if not result.data:
            raise Exception("Failed to insert chunks into database")

        print(f"Successfully processed {len(chunks)} chunks for document {doc_id}")
        return {"success": True, "chunks_processed": len(chunks)}

    except Exception as e:
        error_msg = str(e)
        print(f"Error in ingestion pipeline: {error_msg}")
        return {"success": False, "error": error_msg}
