"""
ingestion.py - PDF Data Extraction & ChromaDB Vector Store Creation

This module handles the first phase of the RAG pipeline:
1. Load PDF documents from the 'data/' folder
2. Split them into smaller chunks for better retrieval
3. Create embeddings and store them in a local ChromaDB vector database

HOW TO RUN:
    python ingestion.py

WHAT TO MODIFY (for students):
    - DATA_DIR        : Change the folder where your PDFs are stored
    - CHUNK_SIZE      : Larger chunks = more context but less precise retrieval
    - CHUNK_OVERLAP   : More overlap = better continuity between chunks
    - EMBEDDING_MODEL : Try different HuggingFace models (see suggestions below)
    - split_documents(): Try a different chunking strategy (see comments inside)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ==========================================================================
# CONFIGURATION - MODIFY HERE
# ==========================================================================

# Folder where your PDF files are placed
DATA_DIR = "data"

# Folder where ChromaDB will persist the vector database
CHROMA_DB_DIR = "chroma_db"

# --------------------------------------------------------------------------
# CHUNKING SETTINGS - Students: experiment with these values!
# --------------------------------------------------------------------------
# CHUNK_SIZE: How many characters per chunk.
#   - Smaller (300-500)  = more precise retrieval, but may lose context
#   - Larger  (1000-2000) = more context per chunk, but less precise matching
CHUNK_SIZE = 600

# CHUNK_OVERLAP: How many characters overlap between consecutive chunks.
#   - More overlap (200-500) = better continuity, chunks share more context
#   - Less overlap (0-100)   = less redundancy, but may miss split sentences
CHUNK_OVERLAP = 100

# --------------------------------------------------------------------------
# EMBEDDING MODEL - Students: try swapping this!
# --------------------------------------------------------------------------
# This model runs locally on your machine (FREE, no API key needed).
# Alternatives to try:
#   "all-MiniLM-L6-v2"           - Fast, lightweight (default, ~80MB)
#   "all-MiniLM-L12-v2"          - Slightly better quality, still fast
#   "all-mpnet-base-v2"          - Best quality from sentence-transformers (~420MB)
#   "paraphrase-MiniLM-L6-v2"    - Good for paraphrase detection
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================================================
# STEP 1: DATA EXTRACTION - Load PDFs
# ==========================================================================

def load_pdfs(data_dir: str) -> list:
    """
    Load all PDF files from the data directory.

    Each PDF page becomes one LangChain Document object with:
    - page_content: the extracted text
    - metadata: {"source": "data/file.pdf", "page": 0}  <-- used for citations!
    """
    documents = []

    # Create the data folder if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}/' folder. Please add PDF files there and re-run.")
        return documents

    # Find all PDF files in the directory
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{data_dir}/'. Please add some and re-run.")
        return documents

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"  Loading: {pdf_file}")

        # PyPDFLoader extracts text page-by-page from the PDF
        # Each page becomes a Document with metadata (source file, page number)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    print(f"  Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    return documents


# ==========================================================================
# STEP 2: CHUNKING - Split documents into smaller pieces
# ==========================================================================

def split_documents(documents: list) -> list:
    """
    Split documents into smaller overlapping chunks for better retrieval.

    WHY CHUNK? Large pages may contain many topics. Smaller chunks let the
    retriever return only the most relevant portion instead of an entire page.

    STUDENTS: Try different chunking strategies below!
    """

    # --------------------------------------------------------------------------
    # MODIFY HERE: Choose a chunking strategy
    # --------------------------------------------------------------------------

    # STRATEGY 1: RecursiveCharacterTextSplitter (DEFAULT)
    # Splits by paragraphs -> sentences -> words, keeping structure intact.
    # This is the most commonly used splitter and a great starting point.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Split priority order
    )

    # STRATEGY 2: CharacterTextSplitter (simpler, less smart)
    # Uncomment below and comment out Strategy 1 to try it:
    #
    # from langchain_text_splitters import CharacterTextSplitter
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separator="\n",  # Splits only on newlines
    # )

    # STRATEGY 3: TokenTextSplitter (splits by tokens instead of characters)
    # Better for LLMs since they process tokens, not characters.
    # Uncomment below to try it:
    #
    # from langchain_text_splitters import TokenTextSplitter
    # text_splitter = TokenTextSplitter(
    #     chunk_size=200,       # 200 tokens per chunk
    #     chunk_overlap=50,     # 50 token overlap
    # )

    # --------------------------------------------------------------------------

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# ==========================================================================
# STEP 3: CREATE VECTOR DB - Embed chunks and store in ChromaDB
# ==========================================================================

def create_vector_store(chunks: list) -> Chroma:
    """
    Convert text chunks into vector embeddings and store in ChromaDB.

    Each chunk is transformed into a numerical vector using the embedding model.
    These vectors are stored locally in ChromaDB for fast similarity search
    during the retrieval phase of the RAG agent.
    """
    # Initialize the embedding model (runs locally, no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create the ChromaDB vector store on disk
    # This stores both the vectors AND the original text + metadata
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    print(f"  Vector store created at '{CHROMA_DB_DIR}/' with {len(chunks)} vectors.")
    return vector_store


# ==========================================================================
# MAIN INGESTION PIPELINE: Extract -> Chunk -> Vectorize
# ==========================================================================

def run_ingestion() -> Chroma | None:
    """
    Run the full ingestion pipeline.
    Returns the ChromaDB vector store if successful, None otherwise.
    """
    print("=" * 50)
    print("STEP 1: DATA EXTRACTION (PDF Loading)")
    print("=" * 50)
    documents = load_pdfs(DATA_DIR)
    if not documents:
        return None

    print()
    print("=" * 50)
    print("STEP 2: CHUNKING (Text Splitting)")
    print("=" * 50)
    chunks = split_documents(documents)

    print()
    print("=" * 50)
    print("STEP 3: CREATE VECTOR DB (ChromaDB)")
    print("=" * 50)
    vector_store = create_vector_store(chunks)

    print("\nIngestion complete!")
    return vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_ingestion()
