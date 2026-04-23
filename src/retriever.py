"""
Retriever module for querying ChromaDB and synthesizing answers.
Handles query embedding, retrieval, and LLM synthesis.
"""
from typing import List, Dict, Optional
from src.processors.embeddings import EmbeddingsModel
from src.vector_db.chromadb_manager import ChromaDBManager
from src.llm.groq_client import GroqClient
from src.llm.fallback_llm import FallbackLLM
from src.config import TOP_K_RETRIEVAL


class Retriever:
    """Query ChromaDB and synthesize answers with source attribution."""

    def __init__(self, db_manager: ChromaDBManager = None):
        """
        Initialize retriever.
        
        Args:
            db_manager: ChromaDBManager instance (creates if None)
        """
        self.embeddings_model = EmbeddingsModel()
        self.db_manager = db_manager or ChromaDBManager()
        
        # Initialize LLM clients
        try:
            self.groq_client = GroqClient()
        except ValueError:
            self.groq_client = None
            print("Warning: Groq API not available. Using fallback LLM.")
        
        self.fallback_llm = FallbackLLM()

    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> Dict:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of results to retrieve
            
        Returns:
            Answer with source attribution
        """
        # Embed the question
        query_embedding = self.embeddings_model.embed(question).tolist()
        
        # Retrieve relevant chunks from ChromaDB
        results = self.db_manager.query(query_embedding, top_k=top_k)
        
        # Format retrieved chunks
        retrieved_chunks = self._format_retrieved_chunks(results)
        
        # Synthesize answer using LLM
        answer = self._synthesize_answer(question, retrieved_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks,
            "num_sources": len(retrieved_chunks),
        }

    def _format_retrieved_chunks(self, results: Dict) -> List[Dict]:
        """
        Format retrieved chunks with source attribution.
        
        Args:
            results: Raw ChromaDB query results
            
        Returns:
            List of formatted chunks with source badges and metadata
        """
        formatted = []
        
        if not results.get('ids') or not results['ids'][0]:
            return formatted
        
        # Extract data from results
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, (doc_id, text, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
            modality = metadata.get('modality', 'unknown')
            
            # Create source badge
            source_badge = self._create_source_badge(metadata)
            
            # Extract snippet from text
            snippet = self._extract_snippet(text, length=150)
            
            formatted.append({
                "id": doc_id,
                "modality": modality,
                "source_badge": source_badge,
                "snippet": snippet,
                "full_text": text,
                "metadata": metadata,
                "relevance_score": 1 - distance,  # Convert distance to relevance
                "rank": i + 1,
            })
        
        return formatted

    def _create_source_badge(self, metadata: Dict) -> str:
        """Create a source badge string from metadata."""
        modality = metadata.get('modality', 'text').upper()
        
        if modality == 'TABLE':
            page = metadata.get('page', '?')
            rows = metadata.get('rows', '?')
            return f"[TABLE] Page {page} ({rows} rows)"
        
        elif modality == 'IMAGE':
            page = metadata.get('page', '?')
            return f"[IMAGE] Page {page}"
        
        else:  # TEXT
            page = metadata.get('page', '?')
            return f"[TEXT] Page {page}"

    @staticmethod
    def _extract_snippet(text: str, length: int = 150) -> str:
        """Extract a snippet from text."""
        if len(text) <= length:
            return text
        
        # Try to cut at word boundary
        snippet = text[:length]
        last_space = snippet.rfind(' ')
        if last_space > 0:
            snippet = snippet[:last_space]
        
        return snippet + "..."

    def _synthesize_answer(self, question: str, sources: List[Dict]) -> str:
        """
        Synthesize an answer using LLM.
        
        Args:
            question: User's question
            sources: Retrieved source chunks
            
        Returns:
            Generated answer
        """
        if not sources:
            return "No relevant information found in the documents."
        
        # Build context from sources
        context_parts = []
        for source in sources:
            badge = source['source_badge']
            text = source['full_text']
            context_parts.append(f"{badge}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Try Groq first, fall back if unavailable
        if self.groq_client:
            try:
                return self.groq_client.synthesize_answer(context, question)
            except Exception as e:
                print(f"Groq API error: {e}. Falling back to local LLM.")
        
        # Fall back to local LLM
        if self.fallback_llm.is_available():
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
            return self.fallback_llm.generate(prompt)
        
        # Final fallback: return concatenated sources
        return f"Based on the documents:\n\n" + "\n\n".join([f"• {s['snippet']}" for s in sources])


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    # Query example (requires ingested documents)
    result = retriever.query("What is the main topic?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['num_sources']}")
    for source in result['sources']:
        print(f"  - {source['source_badge']}: {source['snippet']}")
