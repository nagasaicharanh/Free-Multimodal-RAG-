"""
Chunking strategy benchmark module.
Evaluates different chunking strategies on quality, performance, and cost.
"""
import time
import csv
from typing import List, Dict, Tuple
from pathlib import Path
from src.processors.text_processor import TextProcessor
from src.processors.embeddings import EmbeddingsModel
from src.vector_db.chromadb_manager import ChromaDBManager


class ChunkingBenchmark:
    """Benchmark different text chunking strategies."""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.embeddings_model = EmbeddingsModel()
        self.results = []

    def benchmark_strategies(self, test_documents: List[str]) -> Dict:
        """
        Benchmark multiple chunking strategies.
        
        Args:
            test_documents: List of test texts to chunk
            
        Returns:
            Benchmark results and statistics
        """
        strategies = ["recursive", "fixed", "semantic"]
        results = {
            "strategy": [],
            "avg_chunk_size": [],
            "num_chunks": [],
            "embedding_time_ms": [],
            "retrieval_precision": [],
        }
        
        for strategy in strategies:
            print(f"\nBenchmarking {strategy} strategy...")
            
            all_chunks = []
            for doc in test_documents:
                chunks = self.text_processor.chunk_text(doc, strategy=strategy)
                all_chunks.extend([c['text'] for c in chunks])
            
            # Measure embedding time
            start_time = time.time()
            embeddings = self.embeddings_model.embed(all_chunks)
            embedding_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate metrics
            avg_chunk_size = sum(len(c) for c in all_chunks) / len(all_chunks) if all_chunks else 0
            retrieval_precision = self._evaluate_retrieval_precision(all_chunks, embeddings)
            
            results["strategy"].append(strategy)
            results["avg_chunk_size"].append(round(avg_chunk_size, 1))
            results["num_chunks"].append(len(all_chunks))
            results["embedding_time_ms"].append(round(embedding_time, 2))
            results["retrieval_precision"].append(round(retrieval_precision, 3))
            
            print(f"  Chunks: {len(all_chunks)}")
            print(f"  Avg Size: {avg_chunk_size:.1f} chars")
            print(f"  Embedding Time: {embedding_time:.2f}ms")
            print(f"  Retrieval Precision: {retrieval_precision:.3f}")
        
        return results

    def _evaluate_retrieval_precision(self, chunks: List[str], embeddings) -> float:
        """
        Evaluate retrieval precision using similarity-based clustering.
        
        Returns a score between 0 and 1 based on coherence of chunks.
        """
        if len(chunks) < 2:
            return 1.0
        
        # Calculate average similarity between consecutive chunks
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Precision is measured by how consistent chunk boundaries are
        # Higher variance in similarity = better chunk boundaries
        if not similarities:
            return 1.0
        
        avg_sim = sum(similarities) / len(similarities)
        # Normalize to 0-1 scale (lower avg similarity = better chunking)
        precision = 1 - min(avg_sim, 1.0)
        
        return precision

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)

    def save_results_to_csv(self, results: Dict, output_path: str = "results/chunking_benchmark.csv"):
        """
        Save benchmark results to CSV file.
        
        Args:
            results: Benchmark results dictionary
            output_path: Path to save CSV
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Transpose results for CSV format
        num_strategies = len(results["strategy"])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Metric"] + results["strategy"])
            
            # Rows
            metrics = ["num_chunks", "avg_chunk_size", "embedding_time_ms", "retrieval_precision"]
            for metric in metrics:
                row = [metric.replace("_", " ").title()]
                for i in range(num_strategies):
                    row.append(results[metric][i])
                writer.writerow(row)
        
        print(f"\nResults saved to {output_path}")

    def benchmark_and_save(self, test_documents: List[str], output_path: str = "results/chunking_benchmark.csv"):
        """
        Run benchmark and save results.
        
        Args:
            test_documents: List of test texts
            output_path: Output CSV path
        """
        results = self.benchmark_strategies(test_documents)
        self.save_results_to_csv(results, output_path)
        return results


if __name__ == "__main__":
    # Example usage
    benchmark = ChunkingBenchmark()
    
    # Create test documents
    test_docs = [
        """
        This is a sample document for benchmarking.
        It contains multiple paragraphs and sentences.
        
        The second paragraph provides more context.
        We will use this to evaluate chunking strategies.
        
        A good chunking strategy should produce coherent chunks.
        It should also perform efficiently.
        """ * 3,  # Repeat for larger document
    ]
    
    results = benchmark.benchmark_and_save(test_docs)
    print("\nBenchmark Complete!")
