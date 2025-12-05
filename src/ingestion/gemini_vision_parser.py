import google.generativeai as genai
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pandas as pd
from dataclasses import dataclass
import logging
from ..utils import Logger, ConfigManager

logger = Logger.setup_logger(__name__)
config = ConfigManager()

@dataclass
class GeminiVisionExtraction:
    """Extracted content using Gemini Vision"""
    text: str
    extracted_data: Dict[str, Any]
    confidence_score: float
    content_type: str  # text, table, chart, image_description
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "extracted_data": self.extracted_data,
            "confidence_score": self.confidence_score,
            "content_type": self.content_type
        }

class GeminiVisionParser:
    """PDF parsing using Gemini Vision for advanced extraction"""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or config.get("gemini.api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro-vision")
        self.extraction_prompts = self._load_extraction_prompts()
        self.logger = logger
        
    def _load_extraction_prompts(self) -> Dict[str, str]:
        """Load specialized prompts for different content types"""
        return {
            "general_text": """
            Extract all text content from this document page. 
            Preserve formatting, headers, and structure.
            Include page numbers, footnotes, and captions.
            Return the extracted text in a structured format.
            """,
            
            "table_extraction": """
            Extract all tables from this document. For each table:
            1. Provide the table structure with rows and columns
            2. Convert to Markdown format
            3. Include table captions if present
            4. Note any merged cells
            Format your response as:
            TABLE 1: [description]
            [markdown table]
            
            TABLE 2: [description]
            [markdown table]
            """,
            
            "chart_analysis": """
            Analyze charts and graphs in this document:
            1. Describe the chart type (bar, line, pie, etc.)
            2. Extract data points and trends
            3. Note axis labels and units
            4. Summarize key insights
            Format your response as:
            CHART 1: [type]
            [analysis]
            
            CHART 2: [type]
            [analysis]
            """
        }
    
    def extract_with_vision(self, pdf_path: str, page_num: int = 0) -> GeminiVisionExtraction:
        """Extract content from PDF page using Gemini Vision"""
        try:
            # Convert PDF page to image
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            
            # Prepare image for Gemini
            image_bytes = self._image_to_bytes(page_image)
            
            # Generate content
            response = self.model.generate_content([
                self.extraction_prompts["general_text"],
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return GeminiVisionExtraction(
                text=response.text,
                extracted_data={"raw_response": response.text},
                confidence_score=0.95,
                content_type="text"
            )
            
        except Exception as e:
            self.logger.error(f"Gemini Vision extraction error: {e}")
            raise
    
    def extract_tables_with_vision(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """Extract tables using Gemini Vision"""
        try:
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            image_bytes = self._image_to_bytes(page_image)
            
            response = self.model.generate_content([
                self.extraction_prompts["table_extraction"],
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            # Parse table response
            tables = self._parse_table_response(response.text)
            return tables
            
        except Exception as e:
            self.logger.error(f"Table extraction error: {e}")
            return []
    
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
            if section.startswith("TABLE"):
                if current_table:
                    tables.append(current_table)
                
                # Start new table
                lines = section.split('\n')
                description = lines[0].replace("TABLE", "").strip(": ").strip()
                markdown_table = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                
                current_table = {
                    "description": description,
                    "markdown": markdown_table,
                    "parsed": self._markdown_to_dataframe(markdown_table)
                }
            elif current_table:
                # Continue current table
                current_table["markdown"] += '\n' + section
        
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
