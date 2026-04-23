"""
End-to-end ingestion pipeline for processing PDFs and storing in ChromaDB.
Orchestrates text extraction, processing, and embedding.
"""
from typing import List, Dict, Tuple
import hashlib
import time
from src.pdf_parser import PDFParser
from src.processors.text_processor import TextProcessor
from src.processors.table_processor import TableProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.embeddings import EmbeddingsModel
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient
from src.vector_db.chromadb_manager import ChromaDBManager
from src.config import TOP_K_RETRIEVAL


class IngestionPipeline:
    """End-to-end PDF ingestion pipeline."""

    def __init__(self, db_manager: ChromaDBManager = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            db_manager: ChromaDBManager instance (creates if None)
        """
        self.pdf_parser = PDFParser()
        self.text_processor = TextProcessor()
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()
        self.embeddings_model = EmbeddingsModel()
        self.db_manager = db_manager or ChromaDBManager()
        
        # Initialize LLM clients (with error handling for missing API keys)
        try:
            self.groq_client = GroqClient()
        except ValueError:
            self.groq_client = None
            print("Warning: Groq API key not set. Table summarization will be skipped.")
        
        try:
            self.gemini_client = GeminiClient()
        except ValueError:
            self.gemini_client = None
            print("Warning: Google API key not set. Image description will be skipped.")

    def ingest_pdf(self, pdf_path: str, chunking_strategy: str = "recursive") -> Dict:
        """
        Ingest a PDF file into ChromaDB.
        
        Args:
            pdf_path: Path to PDF file
            chunking_strategy: Text chunking strategy ('recursive', 'fixed', or 'semantic')
            
        Returns:
            Ingestion report with statistics
        """
        start_time = time.time()
        report = {
            "pdf_path": pdf_path,
            "chunking_strategy": chunking_strategy,
            "text_chunks": 0,
            "tables": 0,
            "images": 0,
            "total_embeddings": 0,
            "processing_time": 0,
            "errors": [],
        }
        
        try:
            # Extract content from PDF
            print(f"Extracting content from {pdf_path}...")
            extracted = self.pdf_parser.extract_from_pdf(pdf_path)
            
            # Process text chunks
            if extracted.text_chunks:
                self._process_text_chunks(extracted, chunking_strategy, report)
            
            # Process tables
            if extracted.tables:
                self._process_tables(extracted, report)
            
            # Process images
            if extracted.images:
                self._process_images(extracted, report)
            
            report["processing_time"] = time.time() - start_time
            print(f"Ingestion complete: {report['total_embeddings']} embeddings added in {report['processing_time']:.2f}s")
            
        except Exception as e:
            report["errors"].append(str(e))
            print(f"Error during ingestion: {e}")
        
        return report

    def _process_text_chunks(self, extracted, chunking_strategy: str, report: Dict):
        """Process text chunks and add to ChromaDB."""
        print(f"Processing text chunks with {chunking_strategy} strategy...")
        
        all_chunks = []
        for text in extracted.text_chunks:
            chunks = self.text_processor.chunk_text(text, strategy=chunking_strategy)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embeddings_model.embed(texts)
        
        # Prepare documents for ChromaDB
        documents = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            doc_id = self._generate_id(f"text_{i}_{chunk['text'][:50]}")
            documents.append({
                "id": doc_id,
                "embedding": embedding.tolist(),
                "text": chunk['text'],
                "metadata": {
                    "modality": "text",
                    "strategy": chunking_strategy,
                    "size": chunk['size'],
                }
            })
        
        # Add to ChromaDB
        added_ids = self.db_manager.add_documents(documents)
        report["text_chunks"] = len(added_ids)
        report["total_embeddings"] += len(added_ids)

    def _process_tables(self, extracted, report: Dict):
        """Process tables, summarize with LLM, and add to ChromaDB."""
        print(f"Processing {len(extracted.tables)} tables...")
        
        processed_tables = self.table_processor.process_tables(extracted.tables)
        
        documents = []
        for table_dict in processed_tables:
            table_id = self._generate_id(f"table_{table_dict['page']}_{table_dict['table_index']}")
            
            # Summarize table with LLM if available
            summary = ""
            if self.groq_client:
                try:
                    summary = self.groq_client.summarize_table(table_dict['markdown'])
                except Exception as e:
                    print(f"Error summarizing table: {e}")
            
            # Create text representation for embedding
            table_text = f"Table from page {table_dict['page']}\n{summary}\n{table_dict['markdown']}"
            
            # Embed the table text
            embedding = self.embeddings_model.embed(table_text).tolist()
            
            documents.append({
                "id": table_id,
                "embedding": embedding,
                "text": table_text,
                "metadata": {
                    "modality": "table",
                    "page": table_dict['page'],
                    "rows": table_dict['num_rows'],
                    "cols": table_dict['num_cols'],
                    "summary": summary,
                    "markdown": table_dict['markdown'],
                }
            })
        
        added_ids = self.db_manager.add_documents(documents)
        report["tables"] = len(added_ids)
        report["total_embeddings"] += len(added_ids)

    def _process_images(self, extracted, report: Dict):
        """Process images, describe with Gemini, and add to ChromaDB."""
        print(f"Processing {len(extracted.images)} images...")
        
        processed_images = self.image_processor.process_images(extracted.images)
        
        documents = []
        for img_dict in processed_images:
            img_id = self._generate_id(f"image_{img_dict['page']}_{img_dict['image_index']}")
            
            # Describe image with Gemini if available
            description = ""
            if self.gemini_client:
                try:
                    description = self.gemini_client.describe_image(img_dict['base64'])
                except Exception as e:
                    print(f"Error describing image: {e}")
            
            # Create text representation for embedding
            img_text = f"Image from page {img_dict['page']}\n{description}"
            
            # Embed the image description
            embedding = self.embeddings_model.embed(img_text).tolist()
            
            documents.append({
                "id": img_id,
                "embedding": embedding,
                "text": img_text,
                "metadata": {
                    "modality": "image",
                    "page": img_dict['page'],
                    "width": img_dict['width'],
                    "height": img_dict['height'],
                    "description": description,
                    "xref": img_dict['xref'],
                }
            })
        
        added_ids = self.db_manager.add_documents(documents)
        report["images"] = len(added_ids)
        report["total_embeddings"] += len(added_ids)

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()


if __name__ == "__main__":
    # Example usage
    pipeline = IngestionPipeline()
    
    # Ingest a sample PDF (if available)
    sample_pdf = "data/samples/sample.pdf"
    report = pipeline.ingest_pdf(sample_pdf, chunking_strategy="recursive")
    print(f"Ingestion report: {report}")
