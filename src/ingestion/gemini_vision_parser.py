import google.generativeai as genai
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class GeminiVisionExtraction:
    """Extracted content using Gemini Vision"""
    text: str
    extracted_data: Dict[str, Any]
    confidence_score: float
    content_type: str  # text, table, chart, image_description

class GeminiVisionParser:
    """PDF parsing using Gemini Vision for advanced extraction"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-vision"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.extraction_prompts = self._load_extraction_prompts()
        
    def _load_extraction_prompts(self) -> Dict[str, str]:
        """Load specialized prompts for different content types"""
        return {
            "general_text": """
            Extract all text content from this document page. 
            Preserve formatting, headers, and structure.
            Include page numbers, footnotes, and captions.
            """,
            
            "table_extraction": """
            Extract all tables from this document. For each table:
            1. Provide the table structure with rows and columns
            2. Convert to Markdown format
            3. Include table captions if present
            4. Note any merged cells
            """,
            
            "chart_analysis": """
            Analyze charts and graphs in this document:
            1. Describe the chart type (bar, line, pie, etc.)
            2. Extract data points and trends
            3. Note axis labels and units
            4. Summarize key insights
            """,
            
            "document_structure": """
            Analyze the document structure:
            1. Identify sections and subsections
            2. Extract headings and their hierarchy
            3. Identify bullet points and numbered lists
            4. Note references and citations
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
            logger.error(f"Gemini Vision extraction error: {e}")
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
            logger.error(f"Table extraction error: {e}")
            return []
    
    def analyze_charts(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Analyze charts and graphs in PDF"""
        try:
            page_image = self._pdf_page_to_image(pdf_path, page_num)
            image_bytes = self._image_to_bytes(page_image)
            
            response = self.model.generate_content([
                self.extraction_prompts["chart_analysis"],
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return self._parse_chart_analysis(response.text)
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {}
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        """Convert PDF page to PIL Image"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
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
        lines = response_text.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line and ('---' in line or any(c.isalpha() for c in line)):
                in_table = True
                current_table.append(line)
            elif in_table and line.strip() == '':
                # End of table
                if current_table:
                    tables.append({
                        "markdown": '\n'.join(current_table),
                        "rows": len([l for l in current_table if '|' in l]),
                        "parsed": self._markdown_to_dict(current_table)
                    })
                    current_table = []
                    in_table = False
        
        return tables
    
    def _markdown_to_dict(self, markdown_lines: List[str]) -> List[Dict]:
        """Convert markdown table to list of dictionaries"""
        if not markdown_lines or len(markdown_lines) < 2:
            return []
        
        # Parse headers
        headers = [h.strip() for h in markdown_lines[0].split('|')[1:-1]]
        
        # Parse data rows
        data = []
        for line in markdown_lines[2:]:  # Skip header and separator
            if '|' in line:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) == len(headers):
                    row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                    data.append(row_dict)
        
        return data
    
    def _parse_chart_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse chart analysis response"""
        sections = analysis_text.split('\n\n')
        result = {
            "chart_type": "unknown",
            "data_points": [],
            "insights": [],
            "axes": {}
        }
        
        for section in sections:
            if "chart type" in section.lower():
                result["chart_type"] = section.split(':')[-1].strip()
            elif "axis" in section.lower():
                lines = section.split('\n')
                for line in lines:
                    if 'x-axis' in line.lower():
                        result["axes"]["x"] = line.split(':')[-1].strip()
                    elif 'y-axis' in line.lower():
                        result["axes"]["y"] = line.split(':')[-1].strip()
            elif "insight" in section.lower() or "trend" in section.lower():
                result["insights"].append(section.strip())
        
        return result
