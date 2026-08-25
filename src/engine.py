import os
import requests
import json
import time
import re
from typing import List, Dict, Any, Optional
from .vector_store import LocalVectorStore
from .reranker import LocalReranker
from .evaluator import RAGEvaluator
from .agents.router import IntentRouter, QueryIntent
from .config import (
    DEFAULT_LOCAL_ENDPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    INITIAL_RETRIEVAL_K,
    FINAL_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    HYBRID_SEARCH_ENABLED,
    RERANKING_ENABLED,
    EVALUATION_ENABLED
)


SYSTEM_PROMPT_TEMPLATE = """{persona_prompt}

GÖREVİN VE KESİN KURALLARIN:
1. YALNIZCA aşağıda verilen [BAĞLAM (CONTEXT)] içerisindeki doğrulanmış bilgilere dayanarak yanıt ver.
2. Eğer kullanıcının sorusunun cevabı verilen bağlamda yer almıyorsa veya yetersizse, ASLA uydurma/halüsinasyon yapma. Açıkça "Verilen dokümanlarda bu soruya ilişkin yeterli bilgi bulunmamaktadır." şeklinde belirt.
3. Yanıtında ilgili bilgilerin hangi kaynaktan geldiğini belirt (Örn: [Kaynak: dosya_adi.txt]).
4. Açık, net, anlaşılır ve profesyonel bir dil kullan.

[BAĞLAM (CONTEXT)]:
{context}
"""


class RAGEngine:
    """
    Multi-Agent RAG 2.0 Orchestration Engine:
    1. Multi-Agent Intent Routing (Technical / Executive / Deep Research / Grounded QA)
    2. Multi-turn Contextual Query Reformulation
    3. Hybrid Retrieval (Dense Vector + BM25 Sparse + RRF)
    4. Cross-Encoder Re-Ranking Precision Layer
    5. Grounded Context Injection & Local SLM/LLM Inference
    6. RAG Triad & Confidence Score Evaluation
    """
    def __init__(
        self,
        vector_store: LocalVectorStore,
        endpoint_url: str = DEFAULT_LOCAL_ENDPOINT,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_k: int = FINAL_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        self.vector_store = vector_store
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranker = LocalReranker(top_n=top_k)
        self.evaluator = RAGEvaluator()
        self.router = IntentRouter()

    def reformulate_query(self, user_question: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return user_question

        prev_queries = [m["content"] for m in history if m.get("role") == "user"]
        if prev_queries:
            last_query = prev_queries[-1]
            followup_cues = ["bunun", "bununla", "bunu", "onun", "bu", "peki", "ayrıca", "farkı", "nedir", "what about", "how about", "its", "their"]
            q_lower = user_question.lower()
            if any(cue in q_lower for cue in followup_cues):
                clean_prev = " ".join([w for w in re.findall(r"\w+", last_query) if len(w) > 3])
                return f"{clean_prev} {user_question}"

        return user_question

    def query(self, user_question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Executes the complete Multi-Agent RAG pipeline.
        """
        start_time = time.time()

        # Step 1: Intent Routing
        intent, intent_info = self.router.route(user_question)

        # Step 2: Query Reformulation
        standalone_query = self.reformulate_query(user_question, history or [])

        # Step 3: Hybrid Retrieval (Dense + BM25 + RRF)
        retrieval_k = INITIAL_RETRIEVAL_K * intent_info.get("top_k_multiplier", 1)
        candidates = self.vector_store.hybrid_search(
            query=standalone_query,
            top_k=retrieval_k,
            threshold=self.similarity_threshold
        )

        if not candidates:
            eval_metrics = self.evaluator.evaluate(user_question, "", "Yeterli bilgi bulunmamaktadır.", [])
            return {
                "answer": "Yüklenen dokümanlarda bu soruyla eşleşen yeterli bilgi bulunamadı. Lütfen ilgili dokümanları yüklediğinizden veya sorunuzu farklı ifade ettiğinizden emin olun.",
                "sources": [],
                "grounded": False,
                "context": "",
                "latency_seconds": round(time.time() - start_time, 2),
                "evaluation": eval_metrics,
                "intent_info": intent_info,
                "reformulated_query": standalone_query
            }

        # Step 4: Precision Re-Ranking
        if RERANKING_ENABLED:
            retrieved = self.reranker.rerank(standalone_query, candidates)
        else:
            retrieved = candidates[:self.top_k]

        # Step 5: Build Structured Grounded Context
        context_parts = []
        sources = []
        retrieval_scores = []

        for chunk, score in retrieved:
            doc_name = chunk.get("doc_name", "Belge")
            chunk_idx = chunk.get("chunk_index", 0)
            text = chunk.get("text", "")
            cos_sim = chunk.get("cos_sim", score)
            retrieval_scores.append(cos_sim)

            context_parts.append(f"--- [Belge: {doc_name} | Parça: {chunk_idx} | Alaka: %{cos_sim*100:.1f}] ---\n{text}")
            sources.append({
                "doc_name": doc_name,
                "chunk_index": chunk_idx,
                "score": cos_sim,
                "rerank_score": chunk.get("rerank_score", cos_sim),
                "text_snippet": text[:220] + "..." if len(text) > 220 else text,
                "full_text": text
            })

        formatted_context = "\n\n".join(context_parts)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            persona_prompt=intent_info.get("persona_prompt", "Sen yardımcı bir RAG asistanısın."),
            context=formatted_context
        )

        # Step 6: Local Model Inference
        answer = self._generate_answer(system_prompt, standalone_query, retrieved)

        # Step 7: RAG Triad Evaluation
        eval_metrics = self.evaluator.evaluate(
            query=standalone_query,
            retrieved_context=formatted_context,
            generated_answer=answer,
            retrieval_scores=retrieval_scores
        )

        latency = round(time.time() - start_time, 2)

        return {
            "answer": answer,
            "sources": sources,
            "grounded": True,
            "context": formatted_context,
            "latency_seconds": latency,
            "evaluation": eval_metrics,
            "intent_info": intent_info,
            "reformulated_query": standalone_query
        }

    def _generate_answer(self, system_prompt: str, user_question: str, retrieved_chunks: List[Any]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

        return self._local_synthesizer(user_question, retrieved_chunks)

    def _local_synthesizer(self, user_question: str, retrieved_chunks: List[Any]) -> str:
        top_chunk, score = retrieved_chunks[0]
        doc = top_chunk.get("doc_name", "Belge")
        text = top_chunk.get("text", "")
        
        q_stems = [w[:4].lower() for w in re.findall(r"\w+", user_question) if len(w) >= 3]
        text_lower = text.lower()
        matched_stems = [s for s in q_stems if s in text_lower]

        if len(q_stems) > 0 and len(matched_stems) == 0 and score < 0.50:
            return "Verilen dokümanlarda bu soruya ilişkin yeterli veya doğrulanmış bilgi bulunmamaktadır."

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        relevant_lines = lines[:4] if len(lines) >= 4 else lines
        extracted_content = " ".join(relevant_lines)

        response = (
            f"Dokümanlarınızdan elde edilen bilgilere göre:\n\n"
            f"{extracted_content}\n\n"
            f"📍 **Kaynak:** `{doc}` *(Alaka Oranı: %{score*100:.1f})*"
        )
        
        if len(retrieved_chunks) > 1:
            second_chunk, score2 = retrieved_chunks[1]
            doc2 = second_chunk.get("doc_name", "Belge")
            response += f"\n*Ayrıca `{doc2}` belgesinde de tamamlayıcı bilgiler tespit edildi.*"

        return response
