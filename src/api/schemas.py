from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the documents")
    top_k: int = Field(5, description="Number of documents to retrieve")
    prompt_template: str = Field("qa", description="Prompt template to use")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter for document retrieval")

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    context_length: int
    retrieved_docs: int
    model: str
    timestamp: str
    conversation_id: Optional[str]

class DocumentUploadResponse(BaseModel):
    filename: str
    task_id: Optional[str]
    status: str
    message: str
    pages_processed: Optional[int] = None
    chunks_created: Optional[int] = None

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    start_time: str
    end_time: Optional[str] = None

class BatchQueryRequest(BaseModel):
    questions: List[str]
    top_k: int = 5
    prompt_template: str = "qa"

class BatchQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_questions: int
    processing_time: float

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filter: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    vector_store: Optional[Dict[str, Any]] = None
