import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..utils import Logger

logger = Logger.setup_logger(__name__)

@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    text: str
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ChunkingStrategy:
    """Different chunking strategies"""
    
    @staticmethod
    def recursive_chunking(text: str, chunk_size: int = 1000, 
                          chunk_overlap: int = 200) -> List[str]:
        """Recursive character-based chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    
    @staticmethod
    def semantic_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
        """Semantic-aware chunking on sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 1000) -> List[str]:
        """Fixed-size chunking"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

class DocumentChunker:
    """Main chunking class with multiple strategies"""
    
    def __init__(self, strategy: str = "recursive", 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        
    def chunk_document(self, content: Dict[str, Any], 
                      document_id: str) -> List[DocumentChunk]:
        """Chunk a single document"""
        text = content.get('text', '')
        metadata = content.get('metadata', {})
        page_number = content.get('page_number')
        
        self.logger.info(f"Chunking document {document_id}, text length: {len(text)}")
        
        # Choose chunking strategy
        if self.strategy == "semantic":
            chunks = ChunkingStrategy.semantic_chunking(text, self.chunk_size)
        elif self.strategy == "fixed":
            chunks = ChunkingStrategy.fixed_size_chunking(text, self.chunk_size)
        else:  # recursive
            chunks = ChunkingStrategy.recursive_chunking(
                text, self.chunk_size, self.chunk_overlap
            )
            
        # Create DocumentChunk objects
        document_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = self._generate_chunk_id(document_id, idx, chunk_text)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text)
            })
            
            document_chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                metadata=chunk_metadata,
                page_number=page_number,
                chunk_index=idx
            )
            document_chunks.append(document_chunk)
            
        self.logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
        return document_chunks
    
    def _generate_chunk_id(self, document_id: str, 
                          chunk_index: int, 
                          text: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{document_id}_chunk{chunk_index}_{content_hash}"
    
    def batch_chunk_documents(self, documents: Dict[str, Any]) -> Dict[str, List[DocumentChunk]]:
        """Chunk multiple documents"""
        results = {}
        
        for doc_id, content in documents.items():
            try:
                chunks = self.chunk_document(content, doc_id)
                results[doc_id] = chunks
                self.logger.info(f"Chunked {doc_id} into {len(chunks)} chunks")
            except Exception as e:
                self.logger.error(f"Error chunking document {doc_id}: {e}")
                
        return results
