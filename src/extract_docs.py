import os
import re
import pymupdf
import chromadb
from chromadb.config import Settings
from uuid import uuid4
from typing import List
try:
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    epub = None


def extract_text(filepath: str) -> str:
    """Extract raw text from supported file types."""
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".epub":
        return extract_epub(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_pdf(path: str) -> str:
    """Extract text from PDF, page by page."""
    doc = pymupdf.open(path)
    try:
        text = [page.get_text() for page in doc]
        return "\n".join(text)
    finally:
        doc.close()


def extract_epub(path: str) -> str:
    """Extract text from EPUB, with fallback for complex structures."""
    try:
        doc = pymupdf.open(path)
        text = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(text)
    except Exception as e:
        if not EBOOKLIB_AVAILABLE:
            raise IOError(f"EPUB extraction failed with PyMuPDF and ebooklib not available: {e}")
        
        # Fallback to ebooklib for better HTML/chapter handling
        try:
            book = epub.read_epub(path)
            text = []
            for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
                content = item.get_content().decode('utf-8')
                # Basic HTML stripping for clean text
                clean_content = re.sub(r'<[^>]+>', '', content)  # Strip tags
                clean_content = re.sub(r'\s+', ' ', clean_content)  # Normalize whitespace
                text.append(clean_content.strip())
            return "\n\n".join(text)  # Double newlines for chapter-like breaks
        except Exception as fallback_e:
            raise IOError(f"EPUB fallback failed: {fallback_e}")


def chunk_text(text: str, chunk_size: int = 1536, overlap: int = 256) -> List[str]:
    """Chunk text by sentences first, then words for semantic boundaries. Skips empty chunks."""
    if not text.strip():
        return []
    
    # Step 1: Split into sentences using regex (handles . ! ? with lookbehind)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Step 2: Group sentences into chunks by approximate char length
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size and current_chunk:
            # Finalize chunk and start overlap
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            # Overlap: Start new chunk with last sentence (or part of it)
            if len(current_chunk) > 1:
                current_chunk = current_chunk[-1:]  # Overlap last sentence
                current_length = len(current_chunk[0])
            else:
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
    
    # Add final chunk if non-empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Step 3: Fine-tune with word overlap if needed (for dense sections)
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        i = 0
        while i < len(words):
            end = min(i + chunk_size // 4, len(words))  # Rough word estimate
            subchunk = " ".join(words[i:end])
            if subchunk.strip():  # Skip empties
                final_chunks.append(subchunk)
            i = max(end - overlap // 4, i + 1)  # Overlap in words
    
    return final_chunks if final_chunks else [text[:chunk_size]]  # Fallback for tiny texts


def ingest_dir(dir_path: str, collection_name: str = "docs", persist: bool = False, 
               chunk_size: int = 1000, overlap: int = 200, clear_collection: bool = False,
               exclude_patterns: List[str] = None) -> chromadb.Collection:
    """
    Ingest all supported files in a directory into a Chroma collection.
    
    Args:
        dir_path: Path to directory to ingest.
        collection_name: Name of the Chroma collection.
        persist: If True, save to disk; else, in-memory only.
        chunk_size: Approx chars per chunk.
        overlap: Char overlap between chunks.
        clear_collection: If True, clear existing collection before ingest.
        exclude_patterns: List of filename patterns to skip (e.g., ['.secret', 'private*']).
    
    Returns:
        The Chroma collection instance.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Invalid directory: {dir_path}")
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    client = chromadb.EphemeralClient()

    collection = client.get_or_create_collection(name=collection_name)
    
    if clear_collection:
        collection.delete()  # Wipe and recreate
        collection = client.get_or_create_collection(name=collection_name)
    
    chunk_count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if any(pattern in f for pattern in exclude_patterns):
                print(f"Skipped {f} (excluded)")
                continue
                
            filepath = os.path.join(root, f)
            try:
                raw_text = extract_text(filepath)
                if not raw_text.strip():
                    print(f"Skipped empty {f}")
                    continue
                
                chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                chunk_ids = []
                chunk_docs = []
                chunk_metas = []
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Double-check empties
                        chunk_ids.append(str(uuid4()))
                        chunk_docs.append(chunk)
                        chunk_metas.append({
                            "source": filepath,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        })
                        chunk_count += 1
                
                if chunk_docs:
                    collection.add(
                        ids=chunk_ids,
                        documents=chunk_docs,
                        metadatas=chunk_metas
                    )
                    print(f"Ingested {f} ({len(chunks)} chunks)")
                else:
                    print(f"No valid chunks from {f}")
                    
            except (IOError, ValueError) as e:
                print(f"Failed to ingest {f}: {e}")
            except Exception as e:  # Catch-all for unexpected (log and continue)
                print(f"Unexpected error with {f}: {e}")
    
    if persist:
        client.persist()
        print(f"Database persisted at ./chroma_store. Total chunks: {chunk_count}")
    else:
        print(f"Ephemeral database created. Total chunks: {chunk_count}")
    
