import google.generativeai as genai
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from ..utils import Logger, ConfigManager

logger = Logger.setup_logger(__name__)
config = ConfigManager()

class GeminiMultimodalProcessor:
    """Process multimodal content with Gemini"""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or config.get("gemini.api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        
        # Initialize models
        self.vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        
        logger.info("Initialized GeminiMultimodalProcessor")
        
    def extract_with_vision(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Extract content from PDF page using Gemini Vision"""
        try:
            # Convert PDF page to image
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            
            # Prepare image for Gemini
            image_bytes = self._image_to_bytes(page_image)
            
            # Generate content
            response = self.vision_model.generate_content([
                "Extract all text content from this document page. Preserve formatting and structure.",
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return {
                "text": response.text,
                "page_number": page_num,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Gemini Vision extraction error: {e}")
            return {
                "text": "",
                "page_number": page_num,
                "status": "error",
                "error": str(e)
            }
    
    def extract_tables_with_vision(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """Extract tables using Gemini Vision"""
        try:
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            image_bytes = self._image_to_bytes(page_image)
            
            response = self.vision_model.generate_content([
                "Extract all tables from this document. Convert to markdown format.",
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return self._parse_table_response(response.text)
            
        except Exception as e:
            logger.error(f"Table extraction error: {e}")
            return []
    
    def analyze_charts(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Analyze charts and graphs in PDF"""
        try:
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            image_bytes = self._image_to_bytes(page_image)
            
            response = self.vision_model.generate_content([
                "Analyze charts and graphs in this document. Describe chart types and data.",
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return {
                "analysis": response.text,
                "page_number": page_num,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {
                "analysis": "",
                "page_number": page_num,
                "status": "error",
                "error": str(e)
            }
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        """Convert PDF page to PIL Image"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("ppm")
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        
        return img
    
    def _image_to_bytes(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    
    def _parse_table_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's table extraction response"""
        tables = []
        sections = response_text.split('\n\n')
        
        current_table = None
        for section in sections:
            if "table" in section.lower() or "|" in section and "---" in section:
                if current_table:
                    tables.append(current_table)
                
                current_table = {
                    "markdown": section,
                    "rows": section.count('\n') + 1,
                    "parsed": self._markdown_to_dataframe(section)
                }
            elif current_table:
                current_table["markdown"] += '\n\n' + section
        
        if current_table:
            tables.append(current_table)
        
        return tables
    
    def _markdown_to_dataframe(self, markdown: str) -> pd.DataFrame:
        """Convert markdown table to DataFrame"""
        try:
            lines = markdown.strip().split('\n')
            if len(lines) < 2:
                return pd.DataFrame()
            
            # Parse headers
            headers = [h.strip() for h in lines[0].split('|')[1:-1]]
            
            # Parse data rows
            data = []
            for line in lines[2:]:  # Skip separator line
                if '|' in line:
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        data.append(cells)
            
            return pd.DataFrame(data, columns=headers)
        except:
            return pd.DataFrame()