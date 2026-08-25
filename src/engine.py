import os
import requests
import json
from typing import List, Dict, Any
from .vector_store import LocalVectorStore
from .config import (
    DEFAULT_LOCAL_ENDPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD
)


SYSTEM_PROMPT_TEMPLATE = """Sen kullanıcının yerel doküman havuzuna dayalı olarak sorularını yanıtlayan akıllı bir RAG (Retrieval-Augmented Generation) asistanısın.

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
    RAG Orchestration Engine:
    1. Query embedding & top-k semantic search
    2. Context injection & grounded prompt formation
    3. Local SLM/LLM inference with fallback support
    4. Citations & confidence score tracking
    """
    def __init__(
        self,
        vector_store: LocalVectorStore,
        endpoint_url: str = DEFAULT_LOCAL_ENDPOINT,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        self.vector_store = vector_store
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def query(self, user_question: str) -> Dict[str, Any]:
        """Executes full retrieval and generation pipeline."""
        retrieved = self.vector_store.search(
            query=user_question,
            top_k=self.top_k,
            threshold=self.similarity_threshold
        )

        if not retrieved:
            return {
                "answer": "Yüklenen dokümanlarda bu soruyla eşleşen yeterli bilgi bulunamadı. Lütfen ilgili dokümanları yüklediğinizden veya sorunuzu farklı ifade ettiğinizden emin olun.",
                "sources": [],
                "grounded": False,
                "context": ""
            }

        context_parts = []
        sources = []
        for chunk, score in retrieved:
            doc_name = chunk.get("doc_name", "Belge")
            chunk_idx = chunk.get("chunk_index", 0)
            text = chunk.get("text", "")
            context_parts.append(f"--- [Belge: {doc_name} | Parça: {chunk_idx} | Benzerlik: %{score*100:.1f}] ---\n{text}")
            sources.append({
                "doc_name": doc_name,
                "chunk_index": chunk_idx,
                "score": score,
                "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                "full_text": text
            })

        formatted_context = "\n\n".join(context_parts)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=formatted_context)

        answer = self._generate_answer(system_prompt, user_question, retrieved)

        return {
            "answer": answer,
            "sources": sources,
            "grounded": True,
            "context": formatted_context
        }

    def _generate_answer(self, system_prompt: str, user_question: str, retrieved_chunks: List[Any]) -> str:
        """Sends request to local OpenAI-compatible endpoint (Foundry Local / Ollama / LM Studio)."""
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
        """
        Fallback grounded response synthesizer when external LLM server is disconnected.
        Ensures strict grounding and checks question keyword overlap before responding.
        """
        top_chunk, score = retrieved_chunks[0]
        doc = top_chunk.get("doc_name", "Belge")
        text = top_chunk.get("text", "")
        
        # Keyword relevance check against retrieved text
        q_words = [w.lower() for w in user_question.replace("?", "").replace(".", "").split() if len(w) > 3]
        text_lower = text.lower()
        matched_words = [w for w in q_words if w in text_lower]

        # If none of the meaningful question words exist in the chunk, treat as ungrounded
        if len(q_words) > 0 and len(matched_words) == 0 and score < 0.65:
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

