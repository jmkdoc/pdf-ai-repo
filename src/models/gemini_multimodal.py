import google.generativeai as genai
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
import fitz  # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

class GeminiMultimodalProcessor:
    """Process multimodal content with Gemini"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
        # Initialize models
        self.vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        
    def process_pdf_page(self, 
                        pdf_path: str, 
                        page_num: int,
                        extract_tables: bool = True,
                        extract_charts: bool = True,
                        extract_images: bool = False) -> Dict[str, Any]:
        """Process a PDF page with multimodal capabilities"""
        
        results = {
            "text_content": "",
            "tables": [],
            "charts": [],
            "images": [],
            "structure": {}
        }
        
        try:
            # Convert PDF page to images
            page_images = self._pdf_page_to_images(pdf_path, page_num)
            
            for img_idx, image in enumerate(page_images):
                # Process with Gemini Vision
                vision_result = self._process_with_vision(
                    image, 
                    extract_tables, 
                    extract_charts
                )
                
                # Extract text
                results["text_content"] += vision_result.get("text", "") + "\n"
                
                # Extract tables
                if "tables" in vision_result:
                    results["tables"].extend(vision_result["tables"])
                
                # Extract charts
                if "charts" in vision_result:
                    results["charts"].extend(vision_result["charts"])
                
                # Store images if requested
                if extract_images:
                    img_data = self._image_to_base64(image)
                    results["images"].append({
                        "index": img_idx,
                        "data": img_data[:100] + "..." if len(img_data) > 100 else img_data,
                        "dimensions": image.size
                    })
            
            # Analyze document structure
            results["structure"] = self._analyze_structure(results["text_content"])
            
            return results
            
        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            return results
    
    def _process_with_vision(self, 
                            image: Image.Image, 
                            extract_tables: bool,
                            extract_charts: bool) -> Dict[str, Any]:
        """Process image with Gemini Vision"""
        
        # Convert image to bytes
        img_bytes = self._image_to_bytes(image)
        
        # Build prompts based on extraction needs
        prompts = ["Extract all text from this document page."]
        
        if extract_tables:
            prompts.append("Extract any tables in markdown format.")
        
        if extract_charts:
            prompts.append("Describe any charts or graphs, including data points.")
        
        full_prompt = "\n".join(prompts)
        
        try:
            response = self.vision_model.generate_content([
                full_prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ])
            
            return self._parse_vision_response(response.text)
            
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            return {"text": "", "tables": [], "charts": []}
    
    def _pdf_page_to_images(self, pdf_path: str, page_num: int) -> List[Image.Image]:
        """Convert PDF page to list of images (handles multi-column layouts)"""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Get page dimensions
            rect = page.rect
            
            # For multi-column detection, we could split the page
            # For now, return full page image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF to image conversion error: {e}")
        
        return images
    
    def _parse_vision_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini Vision response"""
        result = {
            "text": "",
            "tables": [],
            "charts": []
        }
        
        lines = response_text.split('\n')
        current_section = "text"
        table_lines = []
        chart_descriptions = []
        
        for line in lines:
            if "TABLE" in line.upper() or "|" in line and "---" in line:
                current_section = "table"
                if line.strip():
                    table_lines.append(line)
            elif "CHART" in line.upper() or "GRAPH" in line.upper():
                current_section = "chart"
                chart_descriptions.append(line)
            elif current_section == "table" and line.strip():
                table_lines.append(line)
            elif current_section == "chart" and line.strip():
                chart_descriptions.append(line)
            else:
                current_section = "text"
                result["text"] += line + "\n"
        
        # Process tables
        if table_lines:
            result["tables"] = self._extract_tables_from_lines(table_lines)
        
        # Process charts
        if chart_descriptions:
            result["charts"] = self._extract_charts_from_descriptions(chart_descriptions)
        
        return result
    
    def _extract_tables_from_lines(self, lines: List[str]) -> List[Dict]:
        """Extract tables from text lines"""
        tables = []
        current_table = []
        
        for line in lines:
            if '|' in line:
                current_table.append(line)
            elif current_table:
                # End of table
                table_text = '\n'.join(current_table)
                tables.append({
                    "markdown": table_text,
                    "rows": len([l for l in current_table if '|' in l]),
                    "parsed": self._parse_markdown_table(current_table)
                })
                current_table = []
        
        return tables
    
    def _parse_markdown_table(self, table_lines: List[str]) -> List[Dict]:
        """Parse markdown table to list of dictionaries"""
        if len(table_lines) < 2:
            return []
        
        # Extract headers
        headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
        
        data = []
        for line in table_lines[2:]:  # Skip separator line
            if '|' in line:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) == len(headers):
                    row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                    data.append(row_dict)
        
        return data
    
    def _extract_charts_from_descriptions(self, descriptions: List[str]) -> List[Dict]:
        """Extract chart information from descriptions"""
        charts = []
        
        current_chart = {}
        for line in descriptions:
            if "chart type" in line.lower():
                current_chart["type"] = line.split(":")[-1].strip()
            elif "data" in line.lower():
                current_chart["data"] = line.split(":")[-1].strip()
            elif "trend" in line.lower() or "insight" in line.lower():
                if "insights" not in current_chart:
                    current_chart["insights"] = []
                current_chart["insights"].append(line.strip())
            
            # If we have a complete chart description
            if len(current_chart) >= 2 and line.strip() == "":
                charts.append(current_chart)
                current_chart = {}
        
        if current_chart:
            charts.append(current_chart)
        
        return charts
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure from text"""
        lines = text.split('\n')
        
        structure = {
            "headings": [],
            "sections": [],
            "lists": [],
            "paragraphs": 0
        }
        
        current_heading_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Detect headings
            if stripped and len(stripped) < 100:  # Heuristic for headings
                if stripped.isupper():
                    structure["headings"].append({"text": stripped, "level": 1})
                elif stripped.startswith(('#', '##', '###')):
                    level = line.count('#')
                    structure["headings"].append({
                        "text": stripped.lstrip('#').strip(),
                        "level": level
                    })
            
            # Detect lists
            if stripped.startswith(('-', '*', '•', '○')) or \
               (stripped and stripped[0].isdigit() and '. ' in stripped[:5]):
                structure["lists"].append(stripped)
            
            # Count paragraphs (non-empty lines)
            if stripped and len(stripped) > 50:  # Heuristic for paragraphs
                structure["paragraphs"] += 1
        
        return structure
    
    def _image_to_bytes(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    
    def _image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string"""
        img_bytes = self._image_to_bytes(image, format)
        return base64.b64encode(img_bytes).decode('utf-8')
