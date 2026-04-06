# settings.py

# Centralized configuration for the PDF AI Tool

# API Settings
API_SETTINGS = {
    'api_key': 'your_api_key',  # API Key for the PDF AI tool
    'api_url': 'https://api.example.com',  # Base URL for the API
}

# Model Settings
MODEL_SETTINGS = {
    'model_type': 'gpt-3.5',  # Example model type
    'max_tokens': 150,  # Maximum tokens for responses
}

# Vector Store Configuration
VECTOR_STORE_SETTINGS = {
    'store_type': 'faiss',  # Type of vector store
    'embedding_dimension': 768,  # Dimension of the embeddings
}

# PDF Processing Settings
PDF_PROCESSING_SETTINGS = {
    'max_pages': 50,  # Maximum number of pages to process
    'text_extraction_method': 'ocr',  # Method for text extraction
}

# Training Settings
TRAINING_SETTINGS = {
    'batch_size': 32,  # Batch size for training
    'learning_rate': 0.001,  # Learning rate for the optimizer
}

# Database Configuration
DATABASE_SETTINGS = {
    'db_type': 'sqlite',  # Type of database
    'db_path': 'path/to/database.db',  # Path to the database file
}