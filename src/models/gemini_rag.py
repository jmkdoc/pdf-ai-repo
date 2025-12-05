import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
from ..embedding.vector_store import VectorStoreManager
from ..embedding.gemini_embeddings import GeminiEmbedding

logger = logging.getLogger(__name__)

class GeminiRAGModel:
    """RAG model using Google Gemini"""
    
    def __init__(self, 
                 vector_store: VectorStoreManager,
                 embedding_model: GeminiEmbedding,
                 gemini_api_key: str,
                 model_name: str = "gemini-1.5-pro",
                 generation_config: Optional[Dict] = None,
                 safety_settings: Optional[List] = None):
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Initialize Gemini model
        self.generation_config = generation_config or {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        self.safety_settings = safety_settings or [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        # Initialize chat history
        self.chat_histories = {}
        
        # Load prompts
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load RAG prompts for different use cases"""
        return {
            "qa": """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
3. Be precise and concise
4. Cite specific parts of the context when possible
5. Format your answer with clear paragraphs and bullet points when appropriate

ANSWER:""",
            
            "summarize": """Summarize the following document content:

CONTENT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive summary
2. Highlight key points and findings
3. Maintain the original meaning
4. Keep it under {max_length} words

SUMMARY:""",
            
            "analyze": """Analyze the following document content:

CONTENT:
{context}

ANALYSIS REQUEST: {question}

INSTRUCTIONS:
1. Provide detailed analysis
2. Identify patterns, trends, and insights
3. Support conclusions with evidence from the text
4. Consider multiple perspectives if applicable

ANALYSIS:""",
            
            "compare": """Compare the following documents:

DOCUMENT 1:
{context1}

DOCUMENT 2:
{context2}

COMPARISON REQUEST: {question}

INSTRUCTIONS:
1. Identify similarities and differences
2. Compare structure, content, and key points
3. Provide specific examples from both documents
4. Draw conclusions based on the comparison

COMPARISON:"""
        }
    
    def answer_question(self, 
                       question: str, 
                       top_k: int = 5,
                       prompt_type: str = "qa",
                       conversation_id: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Answer a question using RAG with Gemini"""
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.vector_store.semantic_search(
            query=question,
            embedding_model=self.embedding_model,
            k=top_k,
            filter=kwargs.get('filter')
        )
        
        # 2. Format context
        context = self._format_context(retrieved_docs)
        
        # 3. Get chat history if conversation_id provided
        history = []
        if conversation_id and conversation_id in self.chat_histories:
            history = self.chat_histories[conversation_id]
        
        # 4. Generate prompt
        prompt = self._generate_prompt(
            prompt_type=prompt_type,
            context=context,
            question=question,
            history=history,
            **kwargs
        )
        
        # 5. Generate response
        try:
            response = self.model.generate_content(prompt)
            
            # 6. Extract answer and citations
            answer = response.text
            citations = self._extract_citations(answer, retrieved_docs)
            
            # 7. Update chat history
            if conversation_id:
                self._update_chat_history(
                    conversation_id, 
                    question, 
                    answer
                )
            
            # 8. Prepare response
            result = {
                "answer": answer,
                "sources": self._prepare_sources(retrieved_docs),
                "citations": citations,
                "context_length": len(context),
                "retrieved_docs": len(retrieved_docs),
                "model": self.model.model_name,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def _generate_prompt(self, 
                        prompt_type: str, 
                        context: str, 
                        question: str,
                        history: List[Dict] = None,
                        **kwargs) -> str:
        """Generate prompt based on type"""
        
        base_prompt = self.prompts.get(prompt_type, self.prompts["qa"])
        
        # Format with context and question
        prompt = base_prompt.format(
            context=context,
            question=question,
            **kwargs
        )
        
        # Add conversation history if available
        if history:
            history_text = "\n".join([
                f"User: {h['question']}\nAssistant: {h['answer']}"
                for h in history[-5:]  # Last 5 exchanges
            ])
            prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"
        
        return prompt
    
    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("metadata", {}).get("filename", "Unknown")
            page = doc.get("metadata", {}).get("page_number", "N/A")
            content = doc.get("document", "")
            similarity = 1 - doc.get("distance", 0)
            
            context_parts.append(
                f"[Source: {source}, Page: {page}, Relevance: {similarity:.2%}]\n"
                f"{content}\n"
                f"{'-'*50}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_citations(self, answer: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """Extract citations from answer"""
        citations = []
        
        # Simple citation extraction based on source mentions
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            source = metadata.get("filename", "")
            page = metadata.get("page_number")
            
            # Check if source is mentioned in answer
            if source and source in answer:
                citations.append({
                    "source": source,
                    "page": page,
                    "content_snippet": doc.get("document", "")[:200],
                    "relevance_score": 1 - doc.get("distance", 0)
                })
        
        return citations
    
    def _prepare_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Prepare source information"""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            
            source = {
                "filename": metadata.get("filename", "Unknown"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "similarity_score": 1 - doc.get("distance", 0),
                "content_preview": doc.get("document", "")[:300] + "...",
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in ["filename", "page_number", "chunk_index"]
                }
            }
            sources.append(source)
        
        return sources
    
    def _update_chat_history(self, 
                            conversation_id: str, 
                            question: str, 
                            answer: str):
        """Update conversation history"""
        if conversation_id not in self.chat_histories:
            self.chat_histories[conversation_id] = []
        
        self.chat_histories[conversation_id].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(self.chat_histories[conversation_id]) > 20:
            self.chat_histories[conversation_id] = \
                self.chat_histories[conversation_id][-20:]
    
    def batch_answer(self, 
                    questions: List[str], 
                    top_k: int = 3,
                    **kwargs) -> List[Dict[str, Any]]:
        """Answer multiple questions"""
        results = []
        
        for question in questions:
            try:
                answer = self.answer_question(
                    question=question,
                    top_k=top_k,
                    **kwargs
                )
                results.append({
                    "question": question,
                    "answer": answer["answer"],
                    "sources": answer["sources"]
                })
            except Exception as e:
                logger.error(f"Error answering question '{question[:50]}...': {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "sources": []
                })
        
        return results
    
    def start_conversation(self) -> str:
        """Start a new conversation and return ID"""
        import uuid
        conversation_id = str(uuid.uuid4())
        self.chat_histories[conversation_id] = []
        return conversation_id
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.chat_histories.get(conversation_id, [])
