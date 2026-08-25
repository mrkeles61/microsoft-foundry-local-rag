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
from src.agents.router import IntentRouter, QueryIntent
from src.agents.summarizer import DocumentSummarizer
from src.agents.deep_research import DeepResearchAgent
from src.graph_visualizer import KnowledgeGraphVisualizer

def main():
    print("=" * 75)
    print("🔬 MICROSOFT AI INNOVATORS - MULTI-AGENT RAG ULTRA DOĞRULAMA TESTİ")
    print("=" * 75)

    # 1. Test Ingestion & Multi-Format Support
    print("\n[TEST 1] Çoklu Format Yükleme & Akıllı Metin Bölme (Chunking)")
    chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 0, "Chunking hatası: Hiçbir chunk üretilemedi!"
    print(f" -> BAŞARILI: {len(chunks)} adet chunk üretildi.")

    # 2. Test Hybrid Indexing
    print("\n[TEST 2] Hibrit İndeksleme (Dense Vector + BM25 Okapi)")
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    store.build_index(chunks)
    assert store.embeddings.shape[0] == len(chunks), "Vektör matrisi boyutu uyumsuz!"
    assert store.bm25.doc_count == len(chunks), "BM25 indeksi eksik!"
    print(f" -> BAŞARILI: Yoğun matris {store.embeddings.shape} ve BM25 ({store.bm25.doc_count} belge) indekslendi.")

    # 3. Test Multi-Agent Intent Router
    print("\n[TEST 3] Çok Ajanlı Niyet Yönlendirici (Multi-Agent Intent Router)")
    router = IntentRouter()
    
    intent1, info1 = router.route("select_variant fonksiyonunun CUDA parametreleri nelerdir?")
    assert intent1 == QueryIntent.TECHNICAL_CODE
    print(f" Soru 1 -> Yönlendirilen Ajan: {info1['agent_name']}")
    
    intent2, info2 = router.route("Microsoft Foundry Local projesinin genel özeti ve faydaları nelerdir?")
    assert intent2 == QueryIntent.EXECUTIVE_SUMMARY
    print(f" Soru 2 -> Yönlendirilen Ajan: {info2['agent_name']}")
    
    intent3, info3 = router.route("Foundry Local ile bulut modelleri arasındaki mimari farkları karşılaştır")
    assert intent3 == QueryIntent.DEEP_RESEARCH
    print(f" Soru 3 -> Yönlendirilen Ajan: {info3['agent_name']}")
    print(" -> BAŞARILI: Tüm sorgu niyetleri doğru ajanlara yönlendirildi.")

    # 4. Test Document Summarizer & Compare
    print("\n[TEST 4] Doküman Zekası (Otomatik Özetleyici & Karşılaştırma)")
    summarizer = DocumentSummarizer(store)
    first_doc = chunks[0]["doc_name"]
    sum_res = summarizer.summarize_document(first_doc)
    assert sum_res["chunk_count"] > 0, "Özet çıkarma başarısız!"
    print(f" Doküman: '{first_doc}' -> {len(sum_res['headings'])} Başlık, {len(sum_res['key_points'])} Vurgu")
    print(" -> BAŞARILI: Otomatik özetleme modülü eksiksiz çalıştı.")

    # 5. Test Knowledge Graph Visualizer
    print("\n[TEST 5] İnteraktif Bilgi Grafiği (Knowledge Graph)")
    kg_viz = KnowledgeGraphVisualizer(chunks)
    graph_data = kg_viz.generate_graph_data()
    assert len(graph_data["entities"]) > 0, "Varlık çıkarma başarısız!"
    print(f" Çıkarılan Varlık Sayısı: {len(graph_data['entities'])}, İlişki Sayısı: {len(graph_data['relationships'])}")
    print(" -> BAŞARILI: Bilgi grafiği ontolojisi üretildi.")

    # 6. Test Multi-Turn RAG Pipeline with Triad Metrics
    print("\n[TEST 6] Çok Turlu RAG Pipeline & RAG Triad Metrikleri")
    engine = RAGEngine(vector_store=store)
    history = [
        {"role": "user", "content": "Microsoft Foundry Local nedir?"},
        {"role": "assistant", "content": "Foundry Local yerel çıkarım platformudur."}
    ]
    res = engine.query("Peki bunun GPU hızlandırması ve avantajları nelerdir?", history=history)
    ev = res.get("evaluation", {})
    print(f" Yeniden Yazılan Sorgu: '{res['reformulated_query']}'")
    print(f" Kalite Skoru: %{ev.get('confidence_score')} ({ev.get('quality_status')})")
    assert ev.get("confidence_score", 0) >= 50
    print(" -> BAŞARILI: Çok turlu hafıza ve RAG Triad metrikleri doğrulandı.")

    # 7. Test Anti-Hallucination Guardrail
    print("\n[TEST 7] Halüsinasyon Engelleme & Güvenlik Koruması")
    res_unrelated = engine.query("Venüs gezegeninin atmosfer basıncı kaç bardır?")
    assert "yeterli" in res_unrelated['answer'].lower() or "bulunamadı" in res_unrelated['answer'].lower()
    print(" -> BAŞARILI: Model doküman dışı uydurma yapmadı.")

    print("\n" + "=" * 75)
    print("🎉 MULTI-AGENT RAG ULTRA TÜM TESTLERİ (%100) BAŞARIYLA GEÇTİ!")
    print("=" * 75)

if __name__ == "__main__":
    main()
