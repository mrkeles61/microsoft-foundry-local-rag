import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from .config import SUPPORTED_EXTENSIONS


def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts raw text from TXT, MD, PDF, Python code, JSON, and CSV files.
    """
    suffix = file_path.suffix.lower()
    
    if suffix in [".txt", ".md", ".py"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
            
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                f.seek(0)
                return f.read()
                
    elif suffix == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = [", ".join(row) for row in reader]
            return "\n".join(rows)
    
    elif suffix == ".pdf":
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required to parse PDF documents. Install via `pip install pypdf`.")
        reader = PdfReader(str(file_path))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(f"\n--- [Sayfa {i + 1}] ---\n" + page_text.strip())
        return "\n".join(extracted_pages)
    
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {suffix}. Desteklenenler: {SUPPORTED_EXTENSIONS}")


def split_text_into_chunks(
    text: str,
    doc_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    Splits text into chunks respecting paragraph and sentence boundaries with sliding overlap.
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_idx = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph fits in current chunk
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append({
                    "id": f"{doc_name}_chunk_{chunk_idx}",
                    "doc_name": doc_name,
                    "chunk_index": chunk_idx,
                    "text": current_chunk.strip()
                })
                chunk_idx += 1
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                # Big single paragraph, split with sliding window
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    part = para[i : i + chunk_size].strip()
                    if part:
                        chunks.append({
                            "id": f"{doc_name}_chunk_{chunk_idx}",
                            "doc_name": doc_name,
                            "chunk_index": chunk_idx,
                            "text": part
                        })
                        chunk_idx += 1
                current_chunk = ""

    if current_chunk.strip():
        chunks.append({
            "id": f"{doc_name}_chunk_{chunk_idx}",
            "doc_name": doc_name,
            "chunk_index": chunk_idx,
            "text": current_chunk.strip()
        })

    return chunks


def load_and_chunk_all_documents(
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    Loads all supported documents from the target directory and chunks them.
    """
    all_chunks = []
    
    if not directory.exists():
        return []

    for file_path in sorted(directory.glob("*")):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                content = extract_text_from_file(file_path)
                file_chunks = split_text_into_chunks(
                    content,
                    doc_name=file_path.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"[Loader] {file_path.name} okunurken hata: {e}")
                
    return all_chunks
