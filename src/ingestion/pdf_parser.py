import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from ..utils import Logger, FileManager

logger = Logger.setup_logger(__name__)

@dataclass
class PDFMetadata:
    """Metadata extracted from PDF"""
    filename: str
    file_size: int
    num_pages: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class PDFContent:
    """Structured PDF content"""
    text: str
    page_number: int
    metadata: PDFMetadata
    images: Optional[List[Dict]] = None
    tables: Optional[List[pd.DataFrame]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "text": self.text,
            "page_number": self.page_number,
            "metadata": self.metadata.to_dict(),
            "images": self.images or [],
            "tables_count": len(self.tables) if self.tables else 0
        }
        if self.tables:
            result["tables_preview"] = [
                table.head().to_dict() for table in self.tables[:3]
            ]
        return result

class PDFParser:
    """Advanced PDF parsing with multiple extraction methods"""
    
    def __init__(self, extract_images: bool = False, extract_tables: bool = True):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.logger = logger
        
    def parse_pdf(self, file_path: str) -> List[PDFContent]:
        """Parse PDF using multiple methods for robustness"""
        self.logger.info(f"Parsing PDF: {file_path}")
        
        if not FileManager.is_valid_pdf(file_path):
            raise ValueError(f"Invalid PDF file: {file_path}")
        
        contents = []
        
        try:
            # Method 1: PyMuPDF for text and metadata
            contents.extend(self._parse_with_pymupdf(file_path))
            
            # Method 2: pdfplumber for tables
            if self.extract_tables:
                contents = self._extract_tables_with_pdfplumber(file_path, contents)
                
            self.logger.info(f"Successfully parsed {len(contents)} pages from {file_path}")
            return contents
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            raise
    
    def _parse_with_pymupdf(self, file_path: str) -> List[PDFContent]:
        """Extract text and metadata using PyMuPDF"""
        contents = []
        
        try:
            with fitz.open(file_path) as doc:
                metadata = self._extract_metadata(doc, file_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    content = PDFContent(
                        text=text,
                        page_number=page_num + 1,
                        metadata=metadata,
                        images=self._extract_images(page) if self.extract_images else None
                    )
                    contents.append(content)
                    
        except Exception as e:
            self.logger.error(f"PyMuPDF parsing error: {e}")
            
        return contents
    
    def _extract_tables_with_pdfplumber(self, file_path: str, 
                                       existing_contents: List[PDFContent]) -> List[PDFContent]:
        """Extract tables using pdfplumber"""
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, content in enumerate(existing_contents):
                    if i < len(pdf.pages):
                        page = pdf.pages[i]
                        tables = page.extract_tables()
                        
                        if tables:
                            df_tables = []
                            for table in tables:
                                if table and len(table) > 0:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    df_tables.append(df)
                            content.tables = df_tables
                            
        except Exception as e:
            self.logger.error(f"Table extraction error: {e}")
            
        return existing_contents
    
    def _extract_metadata(self, doc: "fitz.Document", file_path: str) -> PDFMetadata:
        """Extract PDF metadata"""
        metadata_dict = doc.metadata
        
        return PDFMetadata(
            filename=os.path.basename(file_path),
            file_size=os.path.getsize(file_path),
            num_pages=len(doc),
            title=metadata_dict.get('title'),
            author=metadata_dict.get('author'),
            subject=metadata_dict.get('subject'),
            creation_date=metadata_dict.get('creationDate'),
            modification_date=metadata_dict.get('modDate'),
            keywords=metadata_dict.get('keywords', '').split(',') if metadata_dict.get('keywords') else None
        )
    
    def _extract_images(self, page: "fitz.Page") -> List[Dict]:
        """Extract images from PDF page"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            if base_image:
                image_data = {
                    "index": img_index,
                    "width": base_image.get('width'),
                    "height": base_image.get('height'),
                    "format": base_image.get('ext'),
                    "size": len(base_image.get('image', b'')),
                    "bbox": None
                }
                images.append(image_data)
                
        return images
    
    def batch_parse(self, directory_path: str) -> Dict[str, List[PDFContent]]:
        """Parse multiple PDFs in a directory"""
        results = {}
        FileManager.ensure_directory(directory_path)
        
        for file_path in Path(directory_path).glob("*.pdf"):
            try:
                self.logger.info(f"Parsing: {file_path.name}")
                contents = self.parse_pdf(str(file_path))
                results[file_path.name] = contents
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path.name}: {e}")
                
        return results
    
    def save_parsed_content(self, contents: List[PDFContent], output_dir: str):
        """Save parsed content to JSON files"""
        FileManager.ensure_directory(output_dir)
        
        for content in contents:
            output_file = os.path.join(
                output_dir, 
                f"{content.metadata.filename}_page{content.page_number}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content.to_dict(), f, indent=2, ensure_ascii=False)
