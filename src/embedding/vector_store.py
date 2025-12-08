import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import pickle
import json
from pathlib import Path
import pandas as pd
from ..utils import Logger, ConfigManager, FileManager

logger = Logger.setup_logger(__name__)
config = ConfigManager()

class VectorStore:
    """Base vector store interface"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
    def add_documents(self, documents: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> None:
        """Add documents to vector store"""
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], 
               k: int = 5, 
               filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search similar documents"""
        raise NotImplementedError
    
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        raise NotImplementedError

class ChromaDBStore(VectorStore):
    """ChromaDB vector store implementation - Updated for new API"""
    
    def __init__(self, collection_name: str = "pdf_documents",
                 persist_directory: str = "./data/vector_store",
                 reset: bool = False):
        super().__init__(collection_name)
        
        FileManager.ensure_directory(persist_directory)
        
        # New ChromaDB API (v0.4+)
        try:
            # Try the new API first
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Using new ChromaDB PersistentClient API")
        except Exception as e:
            logger.warning(f"New API failed, trying legacy: {e}")
            # Fallback to legacy API for older versions
            try:
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_directory
                ))
                logger.info("Using legacy ChromaDB Client API")
            except Exception as e2:
                logger.error(f"Both ChromaDB APIs failed: {e2}")
                raise
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Reset collection: {collection_name}")
            except:
                pass
                
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
    def add_documents(self, documents: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> None:
        """Add documents to ChromaDB"""
        try:
            ids = [doc.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]
            texts = [doc.get("text", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
            # Add with embeddings
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: List[float], 
               k: int = 5, 
               filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    result = {
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "id": results['ids'][0][i] if results['ids'] else f"result_{i}"
                    }
                    formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "type": "chromadb",
                "api_version": "new" if hasattr(self.client, '__class__') and 'PersistentClient' in str(self.client.__class__) else "legacy"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store as fallback"""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        super().__init__(collection_name)
        self.documents = []  # List of {"id", "text", "metadata", "embedding"}
        self.logger = logger
        self.logger.info(f"Initialized SimpleVectorStore: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> None:
        """Add documents to simple store"""
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            self.documents.append({
                "id": doc.get("chunk_id", f"doc_{i}"),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "embedding": emb
            })
        self.logger.info(f"Added {len(documents)} documents to SimpleVectorStore")
    
    def search(self, query_embedding: List[float], 
               k: int = 5, 
               filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Simple cosine similarity search"""
        import numpy as np
        
        if not self.documents:
            return []
        
        results = []
        query_vec = np.array(query_embedding)
        
        for doc in self.documents:
            # Apply filter if provided
            if filter:
                skip = False
                for key, value in filter.items():
                    if key not in doc["metadata"] or doc["metadata"][key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            doc_vec = np.array(doc["embedding"])
            
            # Cosine similarity
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(doc_vec) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            
            distance = 1 - similarity  # Convert to distance
            
            results.append({
                "document": doc["text"],
                "metadata": doc["metadata"],
                "distance": distance,
                "id": doc["id"]
            })
        
        # Sort by distance (ascending) and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "type": "simple_memory"
        }

class VectorStoreManager:
    """Manages multiple vector stores with fallback"""
    
    def __init__(self, store_type: str = "chroma", **kwargs):
        self.store_type = store_type
        
        try:
            if store_type == "chroma":
                self.store = ChromaDBStore(**kwargs)
                logger.info(f"Initialized ChromaDBStore")
            elif store_type == "simple":
                self.store = SimpleVectorStore(**kwargs)
                logger.info(f"Initialized SimpleVectorStore")
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
        except Exception as e:
            logger.warning(f"Failed to initialize {store_type} store, falling back to simple: {e}")
            self.store = SimpleVectorStore(**kwargs)
            self.store_type = "simple"
            
    def batch_embed_and_store(self, documents: List[Dict[str, Any]], 
                              embedding_model) -> None:
        """Embed documents and store in vector store"""
        texts = [doc.get("text", "") for doc in documents]
        
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = embedding_model.embed(texts)
        
        self.store.add_documents(documents, embeddings)
        
    def semantic_search(self, query: str, embedding_model, 
                       k: int = 5, 
                       filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        query_embedding = embedding_model.embed_single(query)
        
        results = self.store.search(query_embedding, k=k, filter=filter)
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.store.get_stats()