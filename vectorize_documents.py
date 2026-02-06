"""
Document Vectorization with Citation Metadata
Processes PDFs and MedlinePlus articles into vector embeddings
"""

import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

VECTORSTORE_DIR = "vector_db_dir"
PDF_DIR = "data"
MEDLINEPLUS_DIR = "data/medlineplus"


def load_medlineplus_articles():
    """Load MedlinePlus articles with metadata"""
    medlineplus_path = Path(MEDLINEPLUS_DIR)
    metadata_file = medlineplus_path / "metadata.json"
    
    if not medlineplus_path.exists():
        print("⚠️ MedlinePlus directory not found. Run fetch_medlineplus.py first.")
        return []
    
    if not metadata_file.exists():
        print("⚠️ MedlinePlus metadata not found.")
        return []
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    documents = []
    
    print(f"📚 Loading {len(metadata_dict)} MedlinePlus articles...")
    
    for filepath_str, meta in metadata_dict.items():
        filepath = Path(filepath_str)
        
        if not filepath.exists():
            print(f"  ⚠️ File not found: {filepath}")
            continue
        
        try:
            # Read content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'title': meta['title'],
                    'url': meta['url'],
                    'source': meta['source'],
                    'organization': meta['organization'],
                    'date_updated': meta['date_updated'],
                    'date_fetched': meta['date_fetched'],
                    'word_count': meta['word_count'],
                    'doc_type': 'medlineplus',
                    'filepath': str(filepath)
                }
            )
            
            documents.append(doc)
            
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
    
    print(f"✅ Loaded {len(documents)} MedlinePlus articles")
    return documents


def load_pdf_documents():
    """Load PDF documents from data directory"""
    pdf_dir = Path(PDF_DIR)
    
    if not pdf_dir.exists():
        print("⚠️ PDF directory not found.")
        return []
    
    # Find all PDFs (excluding medlineplus subdirectory)
    pdf_files = [f for f in pdf_dir.glob("*.pdf")]
    
    if not pdf_files:
        print("⚠️ No PDF files found in data directory.")
        return []
    
    print(f"📄 Loading {len(pdf_files)} PDF documents...")
    
    loader = DirectoryLoader(
        path=str(pdf_dir), 
        glob="*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        if 'source' in doc.metadata:
            doc.metadata['doc_type'] = 'pdf'
            doc.metadata['title'] = Path(doc.metadata['source']).stem
    
    print(f"✅ Loaded {len(documents)} PDF pages")
    return documents


def main():
    """Main vectorization process"""
    
    # Check if vector store already exists
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("✅ Vector store already exists. Skipping embedding...")
        print("   Delete 'vector_db_dir' folder to regenerate embeddings.")
        return
    
    print("="*60)
    print("🚀 Starting document vectorization...")
    print("="*60)
    
    # Load all documents
    medlineplus_docs = load_medlineplus_articles()
    pdf_docs = load_pdf_documents()
    
    all_documents = medlineplus_docs + pdf_docs
    
    if not all_documents:
        print("\n❌ No documents found to vectorize!")
        print("   Run: python fetch_medlineplus.py")
        print("   Or add PDF files to 'data/' directory")
        return
    
    print(f"\n📊 Total documents: {len(all_documents)}")
    print(f"   - MedlinePlus: {len(medlineplus_docs)}")
    print(f"   - PDFs: {len(pdf_docs)}")
    
    # Split documents into chunks
    print(f"\n✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    
    text_chunks = text_splitter.split_documents(all_documents)
    print(f"✅ Created {len(text_chunks)} text chunks")
    
    # Create embeddings
    print(f"\n🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector database
    print(f"💾 Creating vector database...")
    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="medical_knowledge"
    )
    
    print(f"✅ Vector database created at: {VECTORSTORE_DIR}")
    print(f"📦 Total chunks stored: {len(text_chunks)}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"✅ VECTORIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"📚 Documents processed: {len(all_documents)}")
    print(f"📦 Chunks created: {len(text_chunks)}")
    print(f"🗂️ Vector store: {VECTORSTORE_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()