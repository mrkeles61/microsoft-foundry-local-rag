import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LocalVectorStore:
    """
    Lightweight, robust local vector database supporting SentenceTransformers
    with Cosine Similarity search and persistent JSON indexing.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", store_path: Path = None):
        self.model_name = model_name
        self.store_path = store_path
        self.model = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        self._init_embedding_model()

    def _init_embedding_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"[VectorStore] SentenceTransformer yüklendi: {self.model_name}")
            except Exception as e:
                print(f"[VectorStore] Uyarı: Model yüklenirken hata ({e}). Yedek n-gram vektörleme kullanılacak.")
                self.model = None
        else:
            print("[VectorStore] sentence-transformers kütüphanesi bulunamadı. Dahili vektörleme aktif.")
            self.model = None

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Lightweight 256-dimensional subword hashing fallback vectorizer."""
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
        """Encodes texts into normalized embedding vectors."""
        if not texts:
            return np.array([])
        
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        else:
            return self._fallback_embed(texts)

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Indexes chunks and calculates dense vector embeddings."""
        if not chunks:
            self.chunks = []
            self.embeddings = np.array([])
            return

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        print(f"[VectorStore] {len(texts)} metin parçası vektörleştiriliyor...")
        self.embeddings = self.encode(texts)
        print(f"[VectorStore] İndeksleme tamamlandı. Matris boyutu: {self.embeddings.shape}")
        
        if self.store_path:
            self.save(self.store_path)

    def search(self, query: str, top_k: int = 4, threshold: float = 0.20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calculates cosine similarity between query and stored chunk vectors.
        Returns list of (chunk_dict, similarity_score) tuples sorted descending.
        """
        if len(self.chunks) == 0 or self.embeddings.size == 0 or not query.strip():
            return []

        query_vec = self.encode([query])[0]
        # Cosine similarity for normalized vectors is simply dot product
        scores = np.dot(self.embeddings, query_vec)

        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.chunks[idx], score))
                
        return results

    def save(self, file_path: Path):
        """Saves chunks and embeddings to JSON format."""
        data = {
            "model_name": self.model_name,
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings.size > 0 else []
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[VectorStore] Vektör indeksi kaydedildi: {file_path}")

    def load(self, file_path: Path) -> bool:
        """Loads index from disk."""
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
            print(f"[VectorStore] {len(self.chunks)} parça başarıyla yüklendi: {file_path}")
            return True
        except Exception as e:
            print(f"[VectorStore] İndeks yükleme hatası: {e}")
            return False
