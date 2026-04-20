"""
Embeddings module for generating text embeddings using sentence-transformers.
Runs entirely locally, no API calls.
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DEVICE


class EmbeddingsModel:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_MODEL_DEVICE):
        """
        Initialize embeddings model.
        
        Args:
            model_name: Name of sentence-transformers model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings (single or batch)
        """
        if isinstance(texts, str):
            return self.model.encode(texts)
        else:
            return self.model.encode(texts, convert_to_numpy=True)

    def embed_with_metadata(self, texts: List[str], metadata: List[dict] = None) -> List[dict]:
        """
        Generate embeddings with metadata.
        
        Args:
            texts: List of text strings
            metadata: Optional list of metadata dicts
            
        Returns:
            List of dicts with 'text', 'embedding', and 'metadata'
        """
        embeddings = self.embed(texts)
        
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            result = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata[i] if metadata else {},
            }
            results.append(result)
        
        return results

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.embed([text1, text2])
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    # Example usage
    model = EmbeddingsModel()
    
    # Embed single text
    embedding = model.embed("Hello, world!")
    print(f"Embedding shape: {embedding.shape}")
    
    # Embed multiple texts
    texts = ["This is text 1", "This is text 2", "This is text 3"]
    embeddings = model.embed(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Check similarity
    sim = model.similarity("cat", "dog")
    print(f"Similarity(cat, dog): {sim:.3f}")
    
    sim2 = model.similarity("cat", "cat")
    print(f"Similarity(cat, cat): {sim2:.3f}")
