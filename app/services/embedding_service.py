import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """
    Embedding Service supporting both Vertex AI (production) and Google AI Studio (dev).

    Provider is selected via LLM_PROVIDER env var:
    - "vertex": Uses Vertex AI with service account authentication
    - "google_ai_studio": Uses Google AI Studio with API key
    """

    def __init__(self):
        provider = os.getenv("LLM_PROVIDER", "vertex").strip().lower()

        if provider == "vertex":
            self.embeddings = self._init_vertex_ai()
        elif provider == "google_ai_studio":
            self.embeddings = self._init_google_ai_studio()
        else:
            raise ValueError(
                f'Invalid LLM_PROVIDER="{provider}". Must be "vertex" or "google_ai_studio".'
            )

    def _init_vertex_ai(self):
        """Initialize Vertex AI embeddings with service account authentication."""

        print("Initializing Vertex AI embeddings")

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_REGION")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for Vertex AI")
        if not location:
            raise ValueError("GOOGLE_CLOUD_REGION is required for Vertex AI")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is required for Vertex AI")

        embed_model = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004").strip()

        try:
            from langchain_google_vertexai import VertexAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-vertexai is required for Vertex AI. "
                "Install it with: pip install langchain-google-vertexai"
            )

        return VertexAIEmbeddings(
            model_name=embed_model,
            project=project,
            location=location,
        )

    def _init_google_ai_studio(self):
        """Initialize Google AI Studio embeddings with API key authentication."""

        print("Initializing Google AI Studio embeddings")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google AI Studio")

        embed_model = os.getenv("GOOGLE_EMBED_MODEL", "text-embedding-004").strip()

        if not embed_model.startswith("models/"):
            embed_model = f"models/{embed_model}"

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google AI Studio. "
                "Install it with: pip install langchain-google-genai"
            )

        return GoogleGenerativeAIEmbeddings(
            model=embed_model,
            google_api_key=api_key,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def get_embeddings(self):
        """Get the embeddings instance."""
        return self.embeddings
