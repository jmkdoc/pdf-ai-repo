import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import os

class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='cache'):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.embeddings_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_embeddings(self, texts, embeddings):
        for text, embedding in zip(texts, embeddings):
            cache_file = os.path.join(self.cache_dir, f'embedding_{hash(text)}.pkl')
            joblib.dump(embedding, cache_file)
            self.embeddings_cache[text] = cache_file

    def _load_embedding(self, text):
        cache_file = self.embeddings_cache.get(text)
        if cache_file and os.path.exists(cache_file):
            return joblib.load(cache_file)
        return None

    def generate_embeddings(self, texts):
        embeddings = []
        for text in texts:
            cached_embedding = self._load_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embedding = self.model.encode(text)
                embeddings.append(embedding)
                self._cache_embeddings([text], [embedding])
        return np.array(embeddings)

    def calculate_similarity(self, embedding1, embedding2):
        cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return cos_sim

    def batch_process(self, texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.generate_embeddings(batch_texts)
            results.append(embeddings)
        return np.vstack(results) if results else np.array([])