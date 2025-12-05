import pytest
import os
from unittest.mock import Mock, patch
from src.models.gemini_rag import GeminiRAGModel
from src.embedding.gemini_embeddings import GeminiEmbedding

class TestGeminiIntegration:
    
    @pytest.fixture
    def mock_gemini_api(self):
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_instance = Mock()
            mock_instance.generate_content.return_value.text = "Test response"
            mock_model.return_value = mock_instance
            yield mock_model
    
    def test_gemini_embedding_initialization(self):
        """Test Gemini embedding model initialization"""
        embedding = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key="test_key",
            task_type="retrieval_document"
        )
        
        assert embedding.model_name == "models/embedding-001"
        assert embedding.task_type == "retrieval_document"
    
    def test_gemini_rag_answer_question(self, mock_gemini_api):
        """Test RAG model with Gemini"""
        # Mock dependencies
        mock_vector_store = Mock()
        mock_vector_store.semantic_search.return_value = [
            {
                "document": "Test document content",
                "metadata": {"filename": "test.pdf", "page_number": 1},
                "distance": 0.1
            }
        ]
        
        mock_embedding = Mock()
        mock_embedding.embed_single.return_value = [0.1] * 768
        
        # Initialize RAG model
        rag_model = GeminiRAGModel(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding,
            gemini_api_key="test_key"
        )
        
        # Test question answering
        result = rag_model.answer_question("Test question")
        
        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Test response"
    
    def test_conversation_history(self):
        """Test conversation history management"""
        rag_model = GeminiRAGModel(
            vector_store=Mock(),
            embedding_model=Mock(),
            gemini_api_key="test_key"
        )
        
        # Start conversation
        conv_id = rag_model.start_conversation()
        
        # Add messages
        rag_model._update_chat_history(conv_id, "Hello", "Hi there!")
        rag_model._update_chat_history(conv_id, "How are you?", "I'm good!")
        
        # Get history
        history = rag_model.get_conversation_history(conv_id)
        
        assert len(history) == 2
        assert history[0]["question"] == "Hello"
        assert history[1]["question"] == "How are you?"
    
    def test_multimodal_processing(self):
        """Test multimodal processing setup"""
        from src.models.gemini_multimodal import GeminiMultimodalProcessor
        
        processor = GeminiMultimodalProcessor(api_key="test_key")
        
        # Test basic initialization
        assert processor.vision_model is not None
        assert processor.text_model is not None

if __name__ == "__main__":
    pytest.main([__file__])
