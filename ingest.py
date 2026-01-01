import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "put_your_key"

# Configuration
DOCS_DIR = "legal_docs"
INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"

def ingest_documents():
    # Ensure directory exists
    Path(DOCS_DIR).mkdir(exist_ok=True)
    
    # Load PDF documents
    print("Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    
    if not documents:
        print("No documents found in legal_docs folder.")
        return
    
    print(f"Loaded {len(documents)} documents")

    # Text splitting optimized for legal context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,           # larger to avoid cutting clauses
        chunk_overlap=600,         # overlap ensures continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings using high-accuracy HuggingFace model
    print("Creating embeddings with all-mpnet-base-v2...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}  # better similarity search
    )

    # Build vector store
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save index and embeddings
    vectorstore.save_local(INDEX_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"Index successfully saved to {INDEX_PATH}")
    print("RAG knowledge base ready!")

if __name__ == "__main__":
    ingest_documents()
