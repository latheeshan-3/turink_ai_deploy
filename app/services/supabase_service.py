import os
import tempfile
from urllib.parse import urlparse

import requests
from app.configs.supabase import supabase_client
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentDataFetcher:
    """Fetches documents from Supabase database."""

    def __init__(self):
        self.client = supabase_client
        self.table_name = "documents"

    def fetch_document_data_by_id(self, doc_id: str):
        """
        Fetch document metadata and get the access URL from Supabase.
        
        Args:
            doc_id: Document UUID as string
            
        Returns:
            Tuple of (doc_metadata dict, doc_access_url string) or None if not found
        """
        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("id", doc_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                doc_data = result.data[0]
                
                # Extract metadata
                doc_metadata = {
                    "title": doc_data.get("title", ""),
                    "source_type": doc_data.get("source_type", ""),
                    "source_path": doc_data.get("source_path", ""),
                }
                
                # Get the public URL directly from source_path
                # Frontend now stores the full public URL in source_path
                doc_access_url = doc_data.get("source_path")
                if not doc_access_url:
                    raise ValueError(f"Document {doc_id} has no source_path")
                
                return doc_metadata, doc_access_url
            else:
                raise ValueError(f"Document with ID {doc_id} not found")

        except Exception as e:
            print(f"Error fetching document with ID {doc_id}: {e}")
            raise


class SupabaseStorageLoader:
    """Load documents directly from Supabase Storage URLs."""

    def load_from_supabase_url(self, url: str):
        """
        Load a document from a Supabase Storage URL.
        
        Args:
            url: Full Supabase Storage URL
            
        Returns:
            List of LangChain Document objects
        """
        try:
            if not self._is_supabase_storage_url(url):
                raise ValueError("URL is not a valid Supabase Storage URL")

            file_ext = self._get_file_extension_from_url(url)

            if file_ext == ".txt":
                documents = self._load_text_from_url(url)
            elif file_ext == ".pdf":
                documents = self._load_pdf_from_url(url)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            return documents

        except Exception as e:
            print(f"Error loading document from URL {url}: {e}")
            raise

    def _load_text_from_url(self, url: str):
        """Load a text file from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            loader = TextLoader(temp_file_path, encoding="utf-8")
            document = loader.load()

            os.unlink(temp_file_path)

            return document

        except Exception as e:
            print(f"Error loading text file: {e}")
            raise

    def _load_pdf_from_url(self, url: str):
        """Load a PDF file from URL."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            document = loader.load()

            os.unlink(temp_file_path)

            return document

        except Exception as e:
            print(f"Error loading PDF file: {e}")
            raise

    def _is_supabase_storage_url(self, url: str) -> bool:
        """Check if URL is a valid Supabase Storage URL."""
        try:
            parsed = urlparse(url)
            return (
                "supabase.co" in parsed.netloc
                and "/storage/v1/object/public/" in parsed.path
            )
        except Exception:
            return False

    def _get_file_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        path_without_query = path.split("?")[0]

        if path_without_query.endswith(".txt"):
            return ".txt"
        elif path_without_query.endswith(".pdf"):
            return ".pdf"
        elif path_without_query.endswith(".docx"):
            return ".docx"
        elif path_without_query.endswith(".doc"):
            return ".doc"
        else:
            return ".unknown"


# Create singleton instances
supabase_storage_loader = SupabaseStorageLoader()
document_data_fetcher = DocumentDataFetcher()
