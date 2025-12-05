# Add these imports at the top
import google.generativeai as genai
from src.models.gemini_rag import GeminiRAGModel
from src.embedding.gemini_embeddings import GeminiEmbedding
from src.models.gemini_multimodal import GeminiMultimodalProcessor

# Update the startup_event function
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup with Gemini"""
    global vector_store, rag_model, gemini_processor
    
    try:
        # Initialize Gemini embedding model
        gemini_embedding = GeminiEmbedding(
            model_name=settings.gemini.models.embedding,
            api_key=settings.gemini.api_key,
            task_type="retrieval_document"
        )
        
        # Initialize vector store with Gemini embeddings
        vector_store = VectorStoreManager(
            store_type=settings.database.vector_db_type,
            collection_name=settings.database.collection_name,
            persist_directory=settings.paths.vector_store
        )
        
        # Initialize Gemini RAG model
        rag_model = GeminiRAGModel(
            vector_store=vector_store,
            embedding_model=gemini_embedding,
            gemini_api_key=settings.gemini.api_key,
            model_name=settings.gemini.models.text,
            generation_config=settings.gemini.generation_config,
            safety_settings=settings.gemini.safety_settings
        )
        
        # Initialize multimodal processor
        gemini_processor = GeminiMultimodalProcessor(
            api_key=settings.gemini.api_key
        )
        
        logger.info("Gemini-powered application started successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# Add new endpoints for Gemini-specific features
@app.post("/gemini/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Query("general", enum=["general", "charts", "tables", "text"])
):
    """Analyze image content with Gemini Vision"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prepare prompt based on analysis type
        prompts = {
            "general": "Describe this image in detail",
            "charts": "Extract and describe any charts or graphs in this image",
            "tables": "Extract any tables from this image in markdown format",
            "text": "Extract all text from this image"
        }
        
        prompt = prompts.get(analysis_type, prompts["general"])
        
        # Generate response
        model = genai.GenerativeModel('gemini-1.5-pro-vision')
        response = model.generate_content([prompt, image])
        
        return {
            "analysis": response.text,
            "analysis_type": analysis_type,
            "filename": file.filename,
            "model": "gemini-1.5-pro-vision"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Image analysis failed: {str(e)}")

@app.post("/gemini/multimodal-process")
async def multimodal_process_pdf(
    file: UploadFile = File(...),
    page_number: int = Query(0, ge=0),
    extract_tables: bool = Query(True),
    extract_charts: bool = Query(True)
):
    """Process PDF with multimodal capabilities"""
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Process with multimodal processor
        result = gemini_processor.process_pdf_page(
            pdf_path=file_path,
            page_num=page_number,
            extract_tables=extract_tables,
            extract_charts=extract_charts
        )
        
        return {
            "filename": file.filename,
            "page_number": page_number,
            "results": result
        }
        
    except Exception as e:
        raise HTTPException(500, f"Multimodal processing failed: {str(e)}")

@app.post("/gemini/chat")
async def chat_with_context(
    request: QueryRequest,
    conversation_id: Optional[str] = None
):
    """Chat with Gemini using conversation context"""
    try:
        if not conversation_id:
            conversation_id = rag_model.start_conversation()
        
        result = rag_model.answer_question(
            question=request.question,
            top_k=request.top_k,
            prompt_type=request.prompt_template,
            conversation_id=conversation_id
        )
        
        return {
            "conversation_id": conversation_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "history_length": len(rag_model.get_conversation_history(conversation_id))
        }
        
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)}")

@app.get("/gemini/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        history = rag_model.get_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        raise HTTPException(404, f"Conversation not found: {str(e)}")
