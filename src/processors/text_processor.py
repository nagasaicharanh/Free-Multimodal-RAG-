"""
Text processor module for chunking and segmentation.
Uses LangChain's RecursiveCharacterTextSplitter for semantic boundaries.
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextProcessor:
    """Process and chunk narrative text from PDFs."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize text processor with chunking parameters.
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize splitter with semantic awareness
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(self, text: str, strategy: str = "recursive") -> List[Dict]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Raw text to chunk
            strategy: Chunking strategy ('recursive', 'fixed', 'semantic')
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if strategy == "recursive":
            return self._chunk_recursive(text)
        elif strategy == "fixed":
            return self._chunk_fixed(text)
        elif strategy == "semantic":
            return self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_recursive(self, text: str) -> List[Dict]:
        """Chunk text using LangChain's recursive splitter."""
        chunks = self.splitter.split_text(text)
        return [
            {
                "text": chunk,
                "strategy": "recursive",
                "size": len(chunk),
            }
            for chunk in chunks
        ]

    def _chunk_fixed(self, text: str) -> List[Dict]:
        """Chunk text into fixed-size pieces."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "strategy": "fixed",
                    "size": len(chunk),
                })
        return chunks

    def _chunk_semantic(self, text: str) -> List[Dict]:
        """
        Chunk text using sentence-level semantic boundaries.
        Splits on sentence endings while respecting chunk size limits.
        """
        chunks = []
        sentences = text.replace("! ", "! |").replace("? ", "? |").split("|")
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk,
                        "strategy": "semantic",
                        "size": len(current_chunk),
                    })
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk,
                "strategy": "semantic",
                "size": len(current_chunk),
            })
        
        return chunks


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor()
    
    sample_text = """
    This is a sample document with multiple paragraphs.
    
    The first paragraph introduces a topic.
    The second paragraph provides more details.
    """
    
    chunks = processor.chunk_text(sample_text, strategy="recursive")
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({chunk['size']} chars): {chunk['text'][:50]}...")
