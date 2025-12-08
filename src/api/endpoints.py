from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import json
import os
import shutil
from typing import List, Optional, Dict, Any
import uuid

from .schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    ProcessingStatus, BatchQueryRequest, BatchQueryResponse,
    SearchRequest, SearchResponse, HealthResponse
)

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunking import DocumentChunker
from src.embedding.gemini_embeddings import GeminiEmbedding
from src.embedding.vector_store import VectorStoreManager
from src.models.gemini_rag import GeminiRAGModel
from src.models.gemini_multimodal import GeminiMultimodalProcessor
from src.utils import Logger, ConfigManager, FileManager

logger = Logger.setup_logger(__name__)
config = ConfigManager()

# Initialize components
vector_store_manager = None
gemini_embedding = None
gemini_rag = None
gemini_processor = None

# Store processing tasks
processing_tasks = {}

app = FastAPI(
    title="PDF AI Repository API",
    description="API for processing and querying PDF documents using Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_components():
    """Initialize all required components with error handling"""
    global vector_store_manager, gemini_embedding, gemini_rag, gemini_processor
    
    try:
        # Initialize directories
        FileManager.ensure_directory(config.get("paths.raw_pdfs"))
        FileManager.ensure_directory(config.get("paths.vector_store"))
        FileManager.ensure_directory(config.get("paths.logs"))
        
        # Initialize embedding model
        try:
            gemini_embedding = GeminiEmbedding(
                model_name=config.get("gemini.models.embedding"),
                api_key=config.get("gemini.api_key"),
                task_type="retrieval_document"
            )
            logger.info("GeminiEmbedding initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiEmbedding: {e}")
            gemini_embedding = None
        
        # Initialize vector store - with fallback
        try:
            vector_store_manager = VectorStoreManager(
                store_type=config.get("database.vector_db_type", "simple"),
                collection_name=config.get("database.collection_name", "pdf_documents"),
                persist_directory=config.get("paths.vector_store")
            )
            logger.info("VectorStoreManager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize VectorStoreManager: {e}")
            # Fallback to simple store
            vector_store_manager = VectorStoreManager(
                store_type="simple",
                collection_name="pdf_documents_fallback"
            )
            logger.info("Using SimpleVectorStore as fallback")
        
        # Initialize RAG model
        try:
            if gemini_embedding:
                gemini_rag = GeminiRAGModel(
                    vector_store=vector_store_manager,
                    embedding_model=gemini_embedding,
                    gemini_api_key=config.get("gemini.api_key"),
                    model_name=config.get("gemini.models.text")
                )
                logger.info("GeminiRAGModel initialized successfully")
            else:
                logger.warning("Skipping GeminiRAGModel initialization - no embedding model")
                gemini_rag = None
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiRAGModel: {e}")
            gemini_rag = None
        
        # Initialize multimodal processor
        try:
            gemini_processor = GeminiMultimodalProcessor(
                api_key=config.get("gemini.api_key")
            )
            logger.info("GeminiMultimodalProcessor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiMultimodalProcessor: {e}")
            gemini_processor = None
        
        logger.info("All components initialized (some may be in fallback mode)")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Don't raise - allow app to start in limited mode
        pass
        
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    initialize_components()

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    vector_store_info = None
    if vector_store_manager:
        vector_store_info = vector_store_manager.get_collection_stats()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        vector_store=vector_store_info
    )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = True,
    extract_tables: bool = True
):
    """Upload and process a PDF document"""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    try:
        # Save file
        raw_pdfs_dir = config.get("paths.raw_pdfs")
        FileManager.ensure_directory(raw_pdfs_dir)
        
        file_path = os.path.join(raw_pdfs_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create task ID
        task_id = f"task_{datetime.now().timestamp()}_{uuid.uuid4().hex[:8]}"
        
        if process_now:
            # Add to background tasks
            background_tasks.add_task(
                process_pdf_document,
                file_path,
                file.filename,
                task_id,
                extract_tables
            )
            
            processing_tasks[task_id] = ProcessingStatus(
                task_id=task_id,
                status="processing",
                progress=0,
                message="Document uploaded, processing started",
                start_time=datetime.now().isoformat()
            )
            
            return DocumentUploadResponse(
                filename=file.filename,
                task_id=task_id,
                status="processing",
                message="Document uploaded and processing started"
            )
        else:
            return DocumentUploadResponse(
                filename=file.filename,
                task_id=None,
                status="uploaded",
                message="Document uploaded successfully, waiting for processing"
            )
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get processing status of a task"""
    if task_id not in processing_tasks:
        raise HTTPException(404, "Task not found")
    
    return processing_tasks[task_id]

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query documents"""
    try:
        if not gemini_rag:
            raise HTTPException(500, "RAG model not initialized")
        
        result = gemini_rag.answer_question(
            question=request.question,
            top_k=request.top_k,
            prompt_type=request.prompt_template,
            conversation_id=request.conversation_id,
            filter=request.filter
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.post("/batch-query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    """Batch query documents"""
    try:
        results = gemini_rag.batch_answer(
            questions=request.questions,
            top_k=request.top_k,
            prompt_template=request.prompt_template
        )
        
        return BatchQueryResponse(
            results=results,
            total_questions=len(results),
            processing_time=0  # Would calculate actual time
        )
        
    except Exception as e:
        logger.error(f"Batch query error: {e}")
        raise HTTPException(500, f"Batch query failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Semantic search in documents"""
    try:
        if not vector_store_manager or not gemini_embedding:
            raise HTTPException(500, "Vector store or embedding model not initialized")
        
        results = vector_store_manager.semantic_search(
            query=request.query,
            embedding_model=gemini_embedding,
            k=request.limit,
            filter=request.filter
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.get("/documents")
async def list_documents(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List documents in the vector store"""
    try:
        # This would require additional methods in vector store
        # For now, return stats
        stats = vector_store_manager.get_collection_stats() if vector_store_manager else {}
        
        return {
            "documents": [],
            "stats": stats,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the vector store"""
    try:
        if vector_store_manager:
            vector_store_manager.store.delete([document_id])
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(500, "Vector store not initialized")
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")

# Gemini-specific endpoints
@app.post("/gemini/analyze")
async def analyze_with_gemini(
    file: UploadFile = File(...),
    page_number: int = Query(0, ge=0),
    analysis_type: str = Query("general", enum=["general", "tables", "charts"])
):
    """Analyze PDF with Gemini Vision"""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with Gemini Vision
        if analysis_type == "tables":
            result = gemini_processor.extract_tables_with_vision(temp_path, page_number)
        elif analysis_type == "charts":
            result = gemini_processor.analyze_charts(temp_path, page_number)
        else:
            result = gemini_processor.extract_with_vision(temp_path, page_number)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "filename": file.filename,
            "page_number": page_number,
            "analysis_type": analysis_type,
            "result": result.to_dict() if hasattr(result, 'to_dict') else result
        }
        
    except Exception as e:
        raise HTTPException(500, f"Gemini analysis failed: {str(e)}")

@app.post("/gemini/chat")
async def chat_with_context(
    request: QueryRequest,
    conversation_id: Optional[str] = None
):
    """Chat with Gemini using conversation context"""
    try:
        if not conversation_id:
            conversation_id = gemini_rag.start_conversation()
        
        result = gemini_rag.answer_question(
            question=request.question,
            top_k=request.top_k,
            prompt_type=request.prompt_template,
            conversation_id=conversation_id,
            filter=request.filter
        )
        
        return {
            "conversation_id": conversation_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "history_length": len(gemini_rag.get_conversation_history(conversation_id))
        }
        
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)}")

@app.get("/gemini/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        history = gemini_rag.get_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        raise HTTPException(404, f"Conversation not found: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

def process_pdf_document(file_path: str, filename: str, task_id: str, extract_tables: bool):
    """Background task to process PDF document"""
    try:
        logger.info(f"Starting processing for {filename}, task: {task_id}")
        
        # Update task status
        processing_tasks[task_id].progress = 10
        processing_tasks[task_id].message = "Parsing PDF"
        
        # Parse PDF
        parser = PDFParser(extract_tables=extract_tables)
        contents = parser.parse_pdf(file_path)
        
        processing_tasks[task_id].progress = 40
        processing_tasks[task_id].message = "Chunking document"
        
        # Chunk document
        chunker = DocumentChunker(
            strategy="recursive",
            chunk_size=config.get("processing.chunk_size", 1000),
            chunk_overlap=config.get("processing.chunk_overlap", 200)
        )
        
        all_chunks = []
        for i, content in enumerate(contents):
            content_dict = {
                "text": content.text,
                "metadata": {
                    "filename": filename,
                    "page_number": content.page_number,
                    "total_pages": len(contents),
                    "source": file_path,
                    "has_tables": content.tables is not None and len(content.tables) > 0
                }
            }
            
            doc_chunks = chunker.chunk_document(
                content_dict, 
                f"{filename}_page{i+1}"
            )
            all_chunks.extend(doc_chunks)
        
        processing_tasks[task_id].progress = 70
        processing_tasks[task_id].message = "Generating embeddings"
        
        # Convert chunks to dict format
        chunk_dicts = [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "metadata": chunk.metadata
            }
            for chunk in all_chunks
        ]
        
        # Generate embeddings and store in vector database
        vector_store_manager.batch_embed_and_store(chunk_dicts, gemini_embedding)
        
        processing_tasks[task_id].progress = 100
        processing_tasks[task_id].status = "completed"
        processing_tasks[task_id].message = f"Processing completed: {len(chunk_dicts)} chunks created"
        processing_tasks[task_id].end_time = datetime.now().isoformat()
        
        logger.info(f"Processing completed for {filename}: {len(chunk_dicts)} chunks")
        
    except Exception as e:
        logger.error(f"Processing error for {filename}: {e}")
        processing_tasks[task_id].status = "failed"
        processing_tasks[task_id].message = f"Processing failed: {str(e)}"
        processing_tasks[task_id].end_time = datetime.now().isoformat()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.endpoints:app",
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000),
        reload=config.get("app.debug", True)
    )
