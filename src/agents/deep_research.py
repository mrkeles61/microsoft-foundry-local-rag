from typing import List, Dict, Any
from ..vector_store import LocalVectorStore


class DeepResearchAgent:
    """
    Deep Research Subagent:
    Executes multi-step retrieval by breaking down broad inquiries into technical sub-queries,
    retrieving from different parts of the knowledge base, and synthesizing a comprehensive brief.
    """
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store

    def conduct_deep_research(self, topic: str, top_k_per_step: int = 3) -> Dict[str, Any]:
        """
        Executes a 3-step deep research process:
        1. Core definition & overview
        2. Technical architecture & implementation
        3. Benefits, trade-offs & optimization
        """
        sub_queries = [
            f"{topic} nedir genel tanım",
            f"{topic} mimari teknik fonksiyonlar kurulum",
            f"{topic} avantajlar performans optimizasyon"
        ]

        aggregated_sources = []
        seen_chunk_ids = set()

        for sq in sub_queries:
            results = self.vector_store.hybrid_search(sq, top_k=top_k_per_step)
            for chunk, score in results:
                chunk_id = chunk.get("id")
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    aggregated_sources.append({
                        "doc_name": chunk.get("doc_name"),
                        "chunk_index": chunk.get("chunk_index"),
                        "score": score,
                        "text": chunk.get("text")
                    })

        # Synthesize multi-perspective report
        report_sections = []
        for i, src in enumerate(aggregated_sources[:5]):
            report_sections.append(f"**Bölüm {i+1} [{src['doc_name']}]:**\n{src['text']}\n")

        full_research_text = "\n\n".join(report_sections) if report_sections else "Araştırma konusuyla ilgili yeterli derinlikte doküman bulunamadı."

        return {
            "topic": topic,
            "sub_queries_used": sub_queries,
            "total_evidence_pieces": len(aggregated_sources),
            "sources": aggregated_sources[:6],
            "research_brief": full_research_text
        }
