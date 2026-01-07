import os
import re
import pymupdf
import chromadb
from uuid import uuid4
from typing import List
from pathlib import Path
try:
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    epub = None


def extract_text(filepath: str) -> str:
    """Extract raw text from supported file types (.pdf, .txt, .epub)."""
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")

    ext = Path(filepath).suffix.lower()
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
                clean_content = re.sub(r'<[^>]+>', '', content)
                clean_content = re.sub(r'\s+', ' ', clean_content)
                text.append(clean_content.strip())
            return "\n\n".join(text)
        except Exception as fallback_e:
            raise IOError(f"EPUB fallback failed: {fallback_e}")


def chunk_text(text: str, chunk_size: int = 1536, overlap: int = 256) -> List[str]:
    """
    Chunk text into overlapping segments while trying to respect sentence boundaries.
    Returns a list of clean, non-empty chunks.
    """
    if not text.strip():
        return []

    # Split on sentence boundaries (period, question, exclamation followed by whitespace)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1

        if current_length + sentence_len > chunk_size and current_chunk_sentences:
            chunk = " ".join(current_chunk_sentences).strip()
            if chunk:
                chunks.append(chunk)

            # Start new chunk with overlap: carry over the last 1–2 sentences if possible
            overlap_sentences = []
            overlap_length = 0
            for prev_sentence in reversed(current_chunk_sentences):
                if overlap_length + len(prev_sentence) + 1 <= overlap:
                    overlap_sentences.append(prev_sentence)
                    overlap_length += len(prev_sentence) + 1
                else:
                    break
            current_chunk_sentences = list(reversed(overlap_sentences))
            current_length = overlap_length
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_len

    if current_chunk_sentences:
        final_chunk = " ".join(current_chunk_sentences).strip()
        if final_chunk:
            chunks.append(final_chunk)

    if not chunks:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        chunks = [c.strip() for c in chunks if c.strip()]

    return chunks
    
