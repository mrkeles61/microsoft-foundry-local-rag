import json
import os
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer supporting multi-language tokens."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """
    Lightweight, high-performance in-memory BM25 Okapi index.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []

    def fit(self, corpus: List[str]):
        self.doc_count = len(corpus)
        if self.doc_count == 0:
            return

        self.doc_len = []
        self.doc_freqs = []
        df: Dict[str, int] = {}

        for doc in corpus:
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in freqs.keys():
                df[t] = df.get(t, 0) + 1

        self.avgdl = sum(self.doc_len) / self.doc_count if self.doc_count > 0 else 0.0

        # Calculate IDF
        self.idf = {}
        for term, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if self.doc_count == 0 or not query_tokens:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, freqs in enumerate(self.doc_freqs):
                if token in freqs:
                    tf = freqs[token]
                    doc_len = self.doc_len[doc_idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += idf_val * (tf * (self.k1 + 1.0) / denom)
                    
        return scores


class LocalVectorStore:
    """
    Production-grade Hybrid Vector Database:
    1. Dense semantic search (SentenceTransformers / Cosine Distance)
    2. Sparse lexical search (BM25 Okapi)
    3. Reciprocal Rank Fusion (RRF) combiner
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", store_path: Path = None):
        self.model_name = model_name
        self.store_path = store_path
        self.model = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        self.bm25 = BM25Index()
        self._init_embedding_model()

    def _init_embedding_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"[VectorStore] Dense Embedding Modeli yüklendi: {self.model_name}")
            except Exception as e:
                print(f"[VectorStore] Uyarı: Model yüklenirken hata ({e}). Yedek n-gram vektörleme kullanılacak.")
                self.model = None
        else:
            print("[VectorStore] sentence-transformers kütüphanesi bulunamadı. Dahili vektörleme aktif.")
            self.model = None

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(256, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                idx = abs(hash(word)) % 256
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        else:
            return self._fallback_embed(texts)

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Builds both dense vector and sparse BM25 indices."""
        if not chunks:
            self.chunks = []
            self.embeddings = np.array([])
            self.bm25 = BM25Index()
            return

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # 1. Build Dense Embeddings
        print(f"[VectorStore] {len(texts)} metin parçası için yoğun vektörler üretiliyor...")
        self.embeddings = self.encode(texts)
        
        # 2. Build Sparse BM25 Index
        print(f"[VectorStore] BM25 seyrek anahtar kelime indeksi oluşturuluyor...")
        self.bm25.fit(texts)
        
        print(f"[VectorStore] Hibrit indeksleme tamamlandı. Matris boyutu: {self.embeddings.shape}")
        if self.store_path:
            self.save(self.store_path)

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Performs pure semantic vector search."""
        if len(self.chunks) == 0 or self.embeddings.size == 0 or not query.strip():
            return []
        query_vec = self.encode([query])[0]
        scores = np.dot(self.embeddings, query_vec)
        ranked = np.argsort(scores)[::-1]
        return [(idx, float(scores[idx])) for idx in ranked[:top_k]]

    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Performs pure lexical BM25 search."""
        if len(self.chunks) == 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(query)
        ranked = np.argsort(scores)[::-1]
        return [(idx, float(scores[idx])) for idx in ranked[:top_k] if scores[idx] > 0]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        rrf_k: int = 60,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        threshold: float = 0.15
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combines Dense & Sparse retrievals using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum( weight / (k + rank) )
        """
        if len(self.chunks) == 0 or not query.strip():
            return []

        # Get larger candidate pool for fusion
        candidate_k = max(top_k * 2, 10)
        dense_results = self.search_dense(query, top_k=candidate_k)
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        rrf_scores: Dict[int, float] = {}

        # 1. Score Dense Ranks
        for rank, (doc_idx, dense_score) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (dense_weight / (rrf_k + rank + 1))

        # 2. Score Sparse Ranks
        for rank, (doc_idx, bm25_score) in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (sparse_weight / (rrf_k + rank + 1))

        # Sort by final fused RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for doc_idx, score in sorted_docs[:top_k]:
            # Calculate raw semantic similarity for reporting confidence percentage
            if self.embeddings.size > 0:
                query_vec = self.encode([query])[0]
                cos_sim = float(np.dot(self.embeddings[doc_idx], query_vec))
            else:
                cos_sim = score

            if cos_sim >= threshold or doc_idx in [d[0] for d in sparse_results[:2]]:
                chunk = self.chunks[doc_idx].copy()
                chunk["rrf_score"] = score
                chunk["cos_sim"] = cos_sim
                final_results.append((chunk, cos_sim))

        return final_results

    def search(self, query: str, top_k: int = 4, threshold: float = 0.18) -> List[Tuple[Dict[str, Any], float]]:
        """Default search method executing Hybrid Retrieval."""
        return self.hybrid_search(query=query, top_k=top_k, threshold=threshold)

    def save(self, file_path: Path):
        data = {
            "model_name": self.model_name,
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings.size > 0 else []
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[VectorStore] Hibrit indeks kaydedildi: {file_path}")

    def load(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            raw_emb = data.get("embeddings", [])
            if raw_emb:
                self.embeddings = np.array(raw_emb, dtype=np.float32)
            else:
                self.embeddings = np.array([])
            
            # Rebuild BM25 in-memory
            texts = [c["text"] for c in self.chunks]
            self.bm25.fit(texts)
            
            print(f"[VectorStore] {len(self.chunks)} parça ve BM25 indeksi yüklendi: {file_path}")
            return True
        except Exception as e:
            print(f"[VectorStore] İndeks yükleme hatası: {e}")
            return False
