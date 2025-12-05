import google.generativeai as genai
from typing import List, Optional
import logging
import numpy as np
from .embeddings import EmbeddingModel
import backoff
from ..utils import Logger, ConfigManager
import PIL.Image
import io

logger = Logger.setup_logger(__name__)
config = ConfigManager()

class GeminiEmbedding(EmbeddingModel):
    """Google Gemini embedding model"""
    
    def __init__(self, model_name: str = "models/embedding-001", 
                 api_key: Optional[str] = None,
                 task_type: str = "retrieval_document"):
        super().__init__(model_name)
        
        api_key = api_key or config.get("gemini.api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        
        self.model_name = model_name
        self.task_type = task_type
        self._validate_task_type()
        
    def _validate_task_type(self):
        """Validate task type parameter"""
        valid_task_types = [
            "retrieval_query",
            "retrieval_document",
            "semantic_similarity",
            "classification",
            "clustering"
        ]
        
        if self.task_type not in valid_task_types:
            logger.warning(f"Task type {self.task_type} not in valid list. Using retrieval_document.")
            self.task_type = "retrieval_document"
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini API"""
        try:
            embeddings = []
            
            for text in texts:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=self.task_type,
                    title="Document chunk"
                )
                
                if result and 'embedding' in result:
                    embeddings.append(result['embedding'])
                else:
                    logger.error(f"No embedding returned for text: {text[:50]}...")
                    embeddings.append([0.0] * 768)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed texts in batches to handle rate limits"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
            
            try:
                batch_embeddings = self.embed(batch)
                all_embeddings.extend(batch_embeddings)
                
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                fallback_embeddings = [self._random_embedding() for _ in batch]
                all_embeddings.extend(fallback_embeddings)
        
        return all_embeddings
    
    def _random_embedding(self, dimension: int = 768) -> List[float]:
        """Generate random embedding as fallback"""
        return list(np.random.randn(dimension).astype(np.float32))
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model"""
        test_text = "Test"
        embedding = self.embed_single(test_text)
        return len(embedding)

class GeminiMultimodalEmbedding(GeminiEmbedding):
    """Multimodal embedding for text and images"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def embed_image(self, image_path: str) -> List[float]:
        """Embed image using Gemini"""
        try:
            # Load image
            image = PIL.Image.open(image_path)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Embed image
            result = genai.embed_content(
                model=self.model_name,
                content=img_bytes,
                task_type=self.task_type
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return self._random_embedding()
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> List[float]:
        """Combine text and image embeddings"""
        text_embedding = self.embed_single(text)
        
        if image_path:
            image_embedding = self.embed_image(image_path)
            # Simple average combination
            combined = [
                (t + i) / 2 
                for t, i in zip(text_embedding, image_embedding)
            ]
            return combined
        
        return text_embedding