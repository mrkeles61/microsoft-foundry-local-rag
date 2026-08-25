import re
from typing import List, Dict, Any, Tuple
import numpy as np


class LocalReranker:
    """
    Precision Re-Ranking layer:
    Evaluates candidate retrieved chunks with lexical overlap, query-term density,
    and semantic alignment to re-rank Top-K chunks with higher precision.
    """
    def __init__(self, top_n: int = 4):
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-scores candidate chunks and returns top_n items sorted by precision score.
        """
        if not candidates:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        reranked = []

        for chunk, base_score in candidates:
            text = chunk.get("text", "").lower()
            text_tokens = re.findall(r"\w+", text)
            total_tokens = len(text_tokens) if len(text_tokens) > 0 else 1

            # 1. Lexical matching ratio
            matched_query_tokens = [t for t in query_tokens if t in text]
            lexical_coverage = len(matched_query_tokens) / len(query_tokens) if query_tokens else 0.0

            # 2. Term density (frequency of query terms in the chunk)
            matched_frequency = sum(text_tokens.count(t) for t in query_tokens)
            density_score = min(matched_frequency / total_tokens * 10, 1.0)

            # 3. Combined Precision Score
            precision_score = (0.55 * base_score) + (0.30 * lexical_coverage) + (0.15 * density_score)
            
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(precision_score)
            chunk_copy["lexical_coverage"] = float(lexical_coverage)
            
            reranked.append((chunk_copy, float(precision_score)))

        # Sort descending by re-rank score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:self.top_n]
