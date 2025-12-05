import os
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback
from datetime import datetime

class Logger:
    """Unified logging utility"""
    
    @staticmethod
    def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            # File handler
            file_handler = logging.FileHandler('logs/app.log')
            file_handler.setLevel(getattr(logging, level.upper()))
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
        try:
            from config import config as imported_config
            return imported_config
        except ImportError:
            # Fallback configuration
            return {
                "app": {"name": "PDF-AI", "debug": True},
                "paths": {
                    "raw_pdfs": "./data/raw_pdfs/",
                    "vector_store": "./data/vector_store/"
                },
                "gemini": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "models": {"text": "gemini-1.5-pro"}
                }
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value

class FileManager:
    """File system utilities"""
    
    @staticmethod
    def ensure_directory(path: str):
        """Ensure directory exists"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension"""
        return Path(file_path).suffix.lower()[1:]  # Remove dot
    
    @staticmethod
    def is_valid_pdf(file_path: str) -> bool:
        """Check if file is a valid PDF"""
        try:
            import fitz
            with fitz.open(file_path) as doc:
                return len(doc) > 0
        except:
            return False
    
    @staticmethod
    def list_pdfs(directory: str) -> List[str]:
        """List all PDF files in directory"""
        pdf_files = []
        for file_path in Path(directory).glob("*.pdf"):
            if file_path.is_file():
                pdf_files.append(str(file_path))
        return pdf_files

class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def handle_exception(e: Exception, context: str = ""):
        """Handle exception with logging"""
        logger = Logger.setup_logger(__name__)
        error_msg = f"Error in {context}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": str(e), "context": context}
