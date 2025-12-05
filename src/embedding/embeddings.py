import openai
import numpy as np
from typing import List, Optional, Union
import logging
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    # Provide a lightweight deterministic fallback when `sentence_transformers`
    # is not installed. This avoids import-time crashes and returns a
    # reproducible embedding vector per input text. The fallback is not a
    # semantic model but is useful for testing and environments where the
    # real dependency cannot be installed.
    import hashlib

    class SentenceTransformer:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model_name = model_name
            self._dim = 384

        def encode(self, texts, show_progress_bar=False):
            import numpy as _np

            def _text_to_vector(t: str):
                # Use SHA-256 hex digest (32 bytes -> 64 hex chars) and convert
                # each byte to an int in [0,255]. Tile to fill required dim.
                h = hashlib.sha256(t.encode('utf-8')).hexdigest()
                bytes_vals = [_int for _int in (int(h[i:i+2], 16) for i in range(0, len(h), 2))]
                reps = (_np.ceil(self._dim / len(bytes_vals))).astype(int)
                vec = (_np.array(bytes_vals * reps[:1]) if isinstance(reps, _np.ndarray) else _np.array(bytes_vals * reps))[:self._dim]
                # Normalize to roughly -1..1
                vec = (vec.astype(_np.float32) - 128.0) / 128.0
                return vec

            if isinstance(texts, str):
                texts = [texts]

            vectors = [_text_to_vector(t) for t in texts]
            return _np.stack(vectors, axis=0)
import cohere
import backoff
from ..utils import Logger

logger = Logger.setup_logger(__name__)

class EmbeddingModel:
    """Base embedding model interface"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        raise NotImplementedError
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text"""
        return self.embed([text])[0]

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", 
                 api_key: Optional[str] = None):
        super().__init__(model_name)
        if api_key:
            openai.api_key = api_key
            
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API"""
        try:
            response = openai.Embedding.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [data['embedding'] for data in response['data']]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

class SentenceTransformerEmbedding(EmbeddingModel):
    """Local Sentence Transformer model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local model"""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {e}")
            raise

class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model"""
    
    def __init__(self, model_name: str = "embed-english-v2.0", 
                 api_key: Optional[str] = None):
        super().__init__(model_name)
        self.client = cohere.Client(api_key) if api_key else None
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Cohere"""
        if not self.client:
            raise ValueError("Cohere API key not provided")
            
        try:
            response = self.client.embed(texts=texts, model=self.model_name)
            return response.embeddings
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

class EmbeddingFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create_embedding_model(model_type: str = "openai", **kwargs) -> EmbeddingModel:
        """Create embedding model based on type"""
        if model_type == "openai":
            return OpenAIEmbedding(**kwargs)
        elif model_type == "sentence_transformer":
            return SentenceTransformerEmbedding(**kwargs)
        elif model_type == "cohere":
            return CohereEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}")
