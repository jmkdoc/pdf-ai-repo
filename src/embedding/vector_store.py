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
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "pdf_documents",
                 persist_directory: str = "./data/vector_store",
                 reset: bool = False):
        super().__init__(collection_name)
        
        FileManager.ensure_directory(persist_directory)
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Reset collection: {collection_name}")
            except:
                pass
                
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
        
    def add_documents(self, documents: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> None:
        """Add documents to ChromaDB"""
        try:
            ids = [doc.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]
            texts = [doc.get("text", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
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
            for i in range(len(results['documents'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
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
                "dimension": None
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class VectorStoreManager:
    """Manages multiple vector stores"""
    
    def __init__(self, store_type: str = "chroma", **kwargs):
        self.store_type = store_type
        
        if store_type == "chroma":
            self.store = ChromaDBStore(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        logger.info(f"Initialized VectorStoreManager with {store_type}")
            
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
