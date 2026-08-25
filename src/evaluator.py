import re
from typing import List, Dict, Any


class RAGEvaluator:
    """
    RAG Triad & Quality Evaluation Engine:
    Calculates 3 standard benchmarks on every generated response:
    1. Context Relevance: Relevance of retrieved documents to user question.
    2. Groundedness: Faithfulness of the answer to the retrieved context (Hallucination check).
    3. Answer Relevance: How directly the answer addresses the user's question.
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        query: str,
        retrieved_context: str,
        generated_answer: str,
        retrieval_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Computes RAG Triad metrics and overall confidence rating (0-100%).
        """
        # 1. Context Relevance
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        context_relevance = min(max(avg_retrieval_score, 0.0), 1.0)

        # 2. Groundedness (Faithfulness)
        # Check what percentage of key terms/entities in the answer originate directly from context
        answer_words = [w.lower() for w in re.findall(r"\w+", generated_answer) if len(w) > 3]
        context_lower = retrieved_context.lower()
        
        if "yeterli bilgi bulunmamaktadır" in generated_answer.lower():
            groundedness = 1.0  # Perfect faithfulness when declaring unknown
            answer_relevance = 1.0
        else:
            supported_words = [w for w in answer_words if w in context_lower]
            groundedness = len(supported_words) / len(answer_words) if answer_words else 0.85

            # 3. Answer Relevance
            query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
            matched_query_terms = [w for w in query_terms if w in generated_answer.lower()]
            answer_relevance = len(matched_query_terms) / len(query_terms) if query_terms else 0.90

        # Composite Confidence Score
        composite_score = (0.35 * context_relevance) + (0.45 * groundedness) + (0.20 * answer_relevance)
        composite_pct = round(composite_score * 100, 1)

        # Status badge
        if composite_pct >= 75:
            status = "Yüksek Güvenilirlik (High Confidence)"
            badge_color = "#107C41"
        elif composite_pct >= 50:
            status = "Orta Güvenilirlik (Medium Confidence)"
            badge_color = "#F7630C"
        else:
            status = "Düşük Güvenilirlik / Doğrulanamadı"
            badge_color = "#D83B01"

        return {
            "context_relevance_pct": round(context_relevance * 100, 1),
            "groundedness_pct": round(groundedness * 100, 1),
            "answer_relevance_pct": round(answer_relevance * 100, 1),
            "confidence_score": composite_pct,
            "quality_status": status,
            "badge_color": badge_color
        }
