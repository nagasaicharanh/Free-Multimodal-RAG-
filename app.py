"""
Streamlit web application for Free Multimodal RAG.
Single-PDF upload interface with Q&A and source attribution.
"""
import streamlit as st
from pathlib import Path
import tempfile
from src.pipeline import IngestionPipeline
from src.retriever import Retriever
from src.vector_db.chromadb_manager import ChromaDBManager

# Must be the first Streamlit command
st.set_page_config(
    page_title="Free Multimodal RAG",
    page_icon="📄",
    layout="wide",
)


def inject_apple_style():
    """Inject Apple-inspired styling."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f5f7;
            --card: rgba(255, 255, 255, 0.72);
            --border: rgba(255, 255, 255, 0.55);
            --text: #1d1d1f;
            --muted: #6e6e73;
            --accent: #0071e3;
        }
        .stApp {
            background: radial-gradient(circle at top, #ffffff 0%, var(--bg) 45%, #ececf1 100%);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
        }
        .hero {
            padding: 1.2rem 0 0.5rem 0;
        }
        .hero h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.02rem;
            margin-top: 0;
        }
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 500;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .block-container {
            padding-top: 1.4rem !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stExpander"]) {
            border-radius: 14px;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid #d2d2d7;
            background: linear-gradient(180deg, #ffffff 0%, #f4f4f6 100%);
            color: #1d1d1f;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            border-color: #b5b5be;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
            transform: translateY(-1px);
        }
        .source-title {
            font-weight: 600;
            color: var(--text);
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = ChromaDBManager()
    
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = IngestionPipeline(st.session_state.db_manager)
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(st.session_state.db_manager)
    
    if "uploaded_pdf" not in st.session_state:
        st.session_state.uploaded_pdf = None
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def display_header():
    """Display application header."""
    st.markdown(
        """
        <div class="hero">
            <h1>Free Multimodal RAG</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Show key metrics
    col1, col2, col3 = st.columns(3)
    doc_count = st.session_state.db_manager.count()
    
    with col1:
        st.metric("Documents Indexed", doc_count)
    with col2:
        st.metric("Query History", len(st.session_state.query_history))
    with col3:
        st.metric("Free APIs", "2 (Groq + Gemini)")


def handle_pdf_upload():
    """Handle PDF file upload and ingestion."""
    st.markdown("### 📥 Upload PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Chunking strategy selection
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox(
                "Chunking Strategy",
                ["recursive", "fixed", "semantic"],
                help="Choose how to split text into chunks"
            )
        
        # Ingest button
        if st.button("🔄 Ingest PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                report = st.session_state.pipeline.ingest_pdf(tmp_path, chunking_strategy=strategy)
            
            # Display report
            st.success(f"✅ Ingestion Complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Text Chunks", report["text_chunks"])
            with col2:
                st.metric("Tables", report["tables"])
            with col3:
                st.metric("Images", report["images"])
            with col4:
                st.metric("Processing Time (s)", f"{report['processing_time']:.2f}")
            
            # Show any errors
            if report["errors"]:
                st.warning("⚠️ Errors encountered:")
                for error in report["errors"]:
                    st.text(error)
            
            st.session_state.uploaded_pdf = uploaded_file.name


def handle_queries():
    """Handle user questions and display answers."""
    st.markdown("### ❓ Ask Questions")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Enter your question",
            placeholder="What is this document about?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_button and question:
        if st.session_state.db_manager.count() == 0:
            st.warning("⚠️ Please upload and ingest a PDF first.")
        else:
            with st.spinner("Searching..."):
                result = st.session_state.retriever.query(question, top_k=3)
            
            # Display answer
            st.markdown("### 📝 Answer")
            st.info(result["answer"])
            
            # Display sources
            st.markdown("### 📚 Sources")
            
            for i, source in enumerate(result["sources"], 1):
                with st.expander(
                    f"{source['source_badge']} - Relevance: {source['relevance_score']:.1%}",
                    expanded=(i == 1)
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Snippet:** {source['snippet']}")
                        
                        # Show modality-specific info
                        if source['modality'] == 'table':
                            if 'markdown' in source['metadata']:
                                st.markdown("**Table:**")
                                st.code(source['metadata']['markdown'], language="")
                            
                            if 'summary' in source['metadata'] and source['metadata']['summary']:
                                st.markdown(f"**Summary:** {source['metadata']['summary']}")
                        
                        elif source['modality'] == 'image':
                            if 'description' in source['metadata'] and source['metadata']['description']:
                                st.markdown(f"**Description:** {source['metadata']['description']}")
                        
                        # Streamlit doesn't allow nested expanders, so render full text inline.
                        st.markdown("**Full Text:**")
                        st.code(source['full_text'], language="text")
                    
                    with col2:
                        st.metric("Relevance", f"{source['relevance_score']:.0%}")
            
            # Add to history
            st.session_state.query_history.append({
                "question": question,
                "answer": result["answer"],
                "num_sources": result["num_sources"]
            })


def display_sidebar():
    """Display sidebar with options."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Clear database option
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.db_manager.clear_collection()
            st.session_state.query_history = []
            st.success("✅ Data cleared!")
        
        # Stats
        st.markdown("### 📊 Stats")
        st.write(f"**Total Documents:** {st.session_state.db_manager.count()}")
        st.write(f"**Queries Made:** {len(st.session_state.query_history)}")
        
        # Info
        st.markdown("### ℹ️ About")
        st.info(
            "Free Multimodal RAG processes PDFs using:\n"
            "- **Text:** sentence-transformers (local)\n"
            "- **Tables:** Groq API (free tier)\n"
            "- **Images:** Google Gemini 1.5 Flash\n"
            "- **Vector DB:** ChromaDB (local)"
        )


def main():
    """Main application entry point."""
    inject_apple_style()
    initialize_session_state()
    display_sidebar()
    display_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Query", "Chunking Benchmark"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            handle_pdf_upload()
        
        with col2:
            handle_queries()
    
    with tab2:
        st.header("📊 Chunking Strategy Benchmark")
        st.markdown(
            "Evaluate different text chunking strategies for optimal retrieval performance."
        )
        
        if st.button("Run Benchmark", use_container_width=True):
            with st.spinner("Running benchmark..."):
                from src.chunking_benchmark import ChunkingBenchmark
                
                # Create test documents
                test_docs = [
                    "This is a sample document for benchmarking. " * 50,  # ~2500 chars
                ]
                
                benchmark = ChunkingBenchmark()
                results = benchmark.benchmark_and_save(test_docs)
            
            # Display results
            st.success("✅ Benchmark complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            for i, strategy in enumerate(results["strategy"]):
                with st.expander(f"**{strategy.title()}** Strategy"):
                    st.metric("Num Chunks", results["num_chunks"][i])
                    st.metric("Avg Chunk Size", f"{results['avg_chunk_size'][i]} chars")
                    st.metric("Embedding Time", f"{results['embedding_time_ms'][i]}ms")
                    st.metric("Retrieval Precision", f"{results['retrieval_precision'][i]:.1%}")
        
        # Show results file
        results_file = Path("results/chunking_benchmark.csv")
        if results_file.exists():
            st.markdown("### 📈 Latest Results")
            with open(results_file, 'r') as f:
                st.code(f.read(), language="csv")


if __name__ == "__main__":
    main()
