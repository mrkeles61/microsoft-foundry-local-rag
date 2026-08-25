from typing import Dict, Any, List
from ..vector_store import LocalVectorStore


class DocumentSummarizer:
    """
    Document Intelligence Agent:
    1. Automatic Executive Summary generation per document.
    2. Multi-document comparison and contrast analysis.
    """
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store

    def summarize_document(self, doc_name: str) -> Dict[str, Any]:
        """
        Extracts and synthesizes key highlights and executive summary for a document.
        """
        doc_chunks = [c for c in self.vector_store.chunks if c.get("doc_name") == doc_name]
        if not doc_chunks:
            return {
                "doc_name": doc_name,
                "summary": f"`{doc_name}` için indekslenmiş metin parçası bulunamadı.",
                "key_points": [],
                "chunk_count": 0
            }

        combined_text = "\n".join([c["text"] for c in doc_chunks])
        
        # Extract headers and key sentences
        lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
        headings = [l.replace("#", "").strip() for l in lines if l.startswith("#")]
        bullet_points = [l.strip("-* ") for l in lines if l.startswith(("-", "*", "•"))]

        if not bullet_points:
            # Fallback: Take first sentence of each chunk
            bullet_points = [c["text"].split(".")[0].strip() for c in doc_chunks if c["text"]]

        return {
            "doc_name": doc_name,
            "chunk_count": len(doc_chunks),
            "headings": headings[:6],
            "key_points": bullet_points[:6],
            "char_count": sum(len(c["text"]) for c in doc_chunks)
        }

    def compare_documents(self, doc_a: str, doc_b: str) -> Dict[str, Any]:
        """
        Compares two documents based on shared vocabulary, core topics, and technical differences.
        """
        sum_a = self.summarize_document(doc_a)
        sum_b = self.summarize_document(doc_b)

        return {
            "doc_a": sum_a,
            "doc_b": sum_b,
            "comparison_table": [
                {"Özellik": "Doküman Adı", doc_a: doc_a, doc_b: doc_b},
                {"Özellik": "Parça Sayısı (Chunks)", doc_a: str(sum_a["chunk_count"]), doc_b: str(sum_b["chunk_count"])},
                {"Özellik": "Toplam Karakter", doc_a: f"{sum_a.get('char_count', 0):,} kr", doc_b: f"{sum_b.get('char_count', 0):,} kr"},
                {"Özellik": "Ana Başlıklar", doc_a: ", ".join(sum_a["headings"][:3]) or "Genel", doc_b: ", ".join(sum_b["headings"][:3]) or "Genel"}
            ]
        }
