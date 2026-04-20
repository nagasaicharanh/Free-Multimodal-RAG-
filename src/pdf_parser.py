"""
PDF Parser module for extracting text, tables, and images from PDF files.
Uses pdfplumber for tables and text, PyMuPDF (fitz) for images.
"""
import io
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """Container for extracted PDF content."""
    text_chunks: List[str]
    tables: List[Dict]
    images: List[Dict]
    metadata: Dict


class PDFParser:
    """Extract text, tables, and images from PDF files."""

    def __init__(self):
        pass

    def extract_from_pdf(self, pdf_path: str) -> ExtractedContent:
        """
        Extract all modalities from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedContent with text_chunks, tables, images, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text_chunks = self._extract_text(pdf_path)
        tables = self._extract_tables(pdf_path)
        images = self._extract_images(pdf_path)
        metadata = {
            "filename": pdf_path.name,
            "file_size_kb": pdf_path.stat().st_size / 1024,
            "num_pages": self._get_pdf_page_count(pdf_path),
            "num_tables": len(tables),
            "num_images": len(images),
        }

        return ExtractedContent(
            text_chunks=text_chunks,
            tables=tables,
            images=images,
            metadata=metadata,
        )

    def _extract_text(self, pdf_path: Path) -> List[str]:
        """Extract narrative text from PDF using pdfplumber."""
        text_chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        # Store with page reference
                        text_chunks.append(
                            f"[Page {page_num}]\n{text}"
                        )
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text_chunks

    def _extract_tables(self, pdf_path: Path) -> List[Dict]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                    page_tables = page.extract_tables(table_settings=table_settings)
                    
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            # Convert to markdown format
                            if table:
                                markdown = self._table_to_markdown(table)
                                tables.append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "raw": table,
                                    "markdown": markdown,
                                })
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
        return tables

    def _extract_images(self, pdf_path: Path) -> List[Dict]:
        """Extract images from PDF using PyMuPDF (fitz)."""
        images = []
        try:
            pdf_doc = fitz.open(pdf_path)
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf_doc, xref)
                    
                    # Convert to PNG bytes
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        image_bytes = pix.tobytes("png")
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        image_bytes = pix.tobytes("png")
                    
                    images.append({
                        "page": page_num + 1,
                        "image_index": img_index,
                        "bytes": image_bytes,
                        "xref": xref,
                    })
            pdf_doc.close()
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {e}")
        return images

    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        """Convert table to markdown format."""
        if not table:
            return ""
        
        # Use first row as header
        header = table[0]
        rows = table[1:]
        
        # Build markdown
        lines = []
        lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        
        for row in rows:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
        
        return "\n".join(lines)

    @staticmethod
    def _get_pdf_page_count(pdf_path: Path) -> int:
        """Get total page count of PDF."""
        try:
            pdf_doc = fitz.open(pdf_path)
            count = len(pdf_doc)
            pdf_doc.close()
            return count
        except Exception:
            return 0


if __name__ == "__main__":
    # Example usage
    parser = PDFParser()
    
    # Create a sample test
    sample_pdf = Path("data/samples/sample.pdf")
    if sample_pdf.exists():
        result = parser.extract_from_pdf(str(sample_pdf))
        print(f"Metadata: {result.metadata}")
        print(f"Text chunks: {len(result.text_chunks)}")
        print(f"Tables: {len(result.tables)}")
        print(f"Images: {len(result.images)}")
    else:
        print("Sample PDF not found. Create one in data/samples/ to test.")
