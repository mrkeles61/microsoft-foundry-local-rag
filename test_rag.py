import sys
import os
from pathlib import Path

# Enable UTF-8 encoding for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DOCUMENTS_DIR, VECTOR_STORE_PATH
from src.loader import load_and_chunk_all_documents
from src.vector_store import LocalVectorStore
from src.engine import RAGEngine

def main():
    print("=" * 70)
    print("🔬 MICROSOFT AI INNOVATORS - RAG 2.0 İLERİ STANDART DOĞRULAMA TESTİ")
    print("=" * 70)

    # 1. Test Ingestion & Multi-Format Support
    print("\n[TEST 1] Çoklu Format Yükleme & Akıllı Metin Bölme (Chunking)")
    chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 0, "Chunking hatası: Hiçbir chunk üretilemedi!"
    print(f" -> BAŞARILI: {len(chunks)} adet chunk üretildi.")
    for c in chunks[:2]:
        print(f"    * Belge: {c['doc_name']} | Parça #{c['chunk_index']} | Karakter: {len(c['text'])}")

    # 2. Test Hybrid Vector & BM25 Indexing
    print("\n[TEST 2] Hibrit İndeksleme (Dense Vector + BM25 Okapi)")
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    store.build_index(chunks)
    assert store.embeddings.shape[0] == len(chunks), "Vektör matrisi boyutu uyumsuz!"
    assert store.bm25.doc_count == len(chunks), "BM25 indeksi eksik!"
    print(f" -> BAŞARILI: Yoğun matris {store.embeddings.shape} ve BM25 ({store.bm25.doc_count} belge) indekslendi.")

    # 3. Test Hybrid Retrieval & Exact Term Matching
    print("\n[TEST 3] Hibrit Arama & Tam Kod Terimi Eşleşmesi (select_variant)")
    query_keyword = "select_variant fonksiyonu ne işe yarar?"
    hybrid_results = store.hybrid_search(query_keyword, top_k=3)
    assert len(hybrid_results) > 0, "Hibrit arama sonuç döndürmedi!"
    top_chunk, score = hybrid_results[0]
    assert "select_variant" in top_chunk["text"], "BM25 tam eşleşme terimi yakalayamadı!"
    print(f" Soru: '{query_keyword}'")
    print(f" -> BAŞARILI: '{top_chunk['doc_name']}' içinde tam eşleşen kod terimi bulundu. (RRF Skoru: {top_chunk.get('rrf_score', 0):.4f})")

    # 4. Test Multi-Turn Conversation & RAG Triad Evaluation
    print("\n[TEST 4] Çok Turlu Soru Yeniden Yazma (Query Reformulation) & RAG Triad")
    engine = RAGEngine(vector_store=store)
    
    # Simulate conversation history
    history = [
        {"role": "user", "content": "Microsoft Foundry Local nedir?"},
        {"role": "assistant", "content": "Microsoft Foundry Local yerel yapay zeka model çalıştırma platformudur."}
    ]
    followup_q = "Peki bunun GPU hızlandırması ve avantajları nelerdir?"
    res = engine.query(followup_q, history=history)
    
    print(f" Takip Sorusu: '{followup_q}'")
    print(f" Yeniden Yazılan Arama Sorgusu: '{res['reformulated_query']}'")
    print(f" Üretilen Cevap:\n{res['answer']}")
    
    ev = res.get("evaluation", {})
    print(f"\n 📊 RAG Triad Kalite Skorları:")
    print(f"    * Context Relevance: %{ev.get('context_relevance_pct')}")
    print(f"    * Groundedness (Sadakat): %{ev.get('groundedness_pct')}")
    print(f"    * Answer Relevance: %{ev.get('answer_relevance_pct')}")
    print(f"    * Genel Güvenilirlik Skoru: %{ev.get('confidence_score')} -> {ev.get('quality_status')}")
    assert ev.get("confidence_score", 0) >= 50, "Güvenilirlik skoru beklenenden düşük!"

    # 5. Test Anti-Hallucination Guardrail
    print("\n[TEST 5] Halüsinasyon Engelleme & Güvenlik Koruması")
    query_unrelated = "Mars gezegeninde bulunan su miktarı kaç litredir?"
    res_unrelated = engine.query(query_unrelated)
    print(f" Alakasız Soru: '{query_unrelated}'")
    print(f" Cevap: {res_unrelated['answer']}")
    assert "yeterli" in res_unrelated['answer'].lower() or "bulunamadı" in res_unrelated['answer'].lower()
    print(" -> BAŞARILI: Model uydurma yapmadı ve dürüstçe uyardı.")

    print("\n" + "=" * 70)
    print("🎉 TÜM RAG 2.0 İLERİ STANDART TESTLERİ BAŞARIYLA GEÇTİ (%100)")
    print("=" * 70)

if __name__ == "__main__":
    main()
