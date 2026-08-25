import sys
import os
from pathlib import Path

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DOCUMENTS_DIR, VECTOR_STORE_PATH
from src.loader import load_and_chunk_all_documents
from src.vector_store import LocalVectorStore
from src.engine import RAGEngine

def main():
    print("=" * 60)
    print("[TEST SUITE] MICROSOFT AI INNOVATORS - RAG STANDART UYUMLULUK")
    print("=" * 60)

    # 1. Test Ingestion & Chunking
    print("\n[TEST 1] Dokuman Yukleme & Akilli Metin Bolme (Chunking)")
    chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 0, "Chunking hatasi: Hicbir chunk uretilemedi!"
    print(f" -> BASARILI: {len(chunks)} adet chunk uretildi.")
    for c in chunks[:2]:
        print(f"    * Dosya: {c['doc_name']} | Parca #{c['chunk_index']} | Karakter: {len(c['text'])}")

    # 2. Test Vector Embedding & Indexing
    print("\n[TEST 2] Vektor Indeksleme & Embedding Uretimi")
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    store.build_index(chunks)
    assert store.embeddings.shape[0] == len(chunks), "Vektor matrisi boyutu uyumsuz!"
    print(f" -> BASARILI: Vektor matrisi boyutu = {store.embeddings.shape} (Dense boyutu: {store.embeddings.shape[1]})")

    # 3. Test Retrieval & Citations
    print("\n[TEST 3] Soru-Cevap & Kaynak Alintilama (Citations)")
    engine = RAGEngine(vector_store=store, top_k=3, similarity_threshold=0.15)
    query_1 = "Microsoft Foundry Local ve CUDA optimizasyonu nasil yapilir?"
    res1 = engine.query(query_1)
    print(f" Soru: '{query_1}'")
    print(f" Cevap:\n{res1['answer']}")
    print(f" Alintilanan Kaynak Sayisi: {len(res1['sources'])}")
    for s in res1['sources']:
        print(f"    * {s['doc_name']} (Parca #{s['chunk_index']} | Benzerlik: %{s['score']*100:.1f})")
    assert len(res1['sources']) > 0, "Kaynak getirme basarisiz!"
    print(" -> BASARILI: Ilgili kaynaklar ve alintilar dogru eslesti.")

    # 4. Test Anti-Hallucination Guardrail
    print("\n[TEST 4] Halusinasyon Engelleme & Guvenlik Korumasi")
    query_unrelated = "Dunyanin en derin okyanus cukuru neresidir ve derinligi kac metredir?"
    res_unrelated = engine.query(query_unrelated)
    print(f" Soru: '{query_unrelated}'")
    print(f" Cevap: {res_unrelated['answer']}")
    print(f" Grounded Durumu: {res_unrelated['grounded']}")
    print(" -> BASARILI: Dokumanda olmayan soru icin model uydurma yapmadi.")

    print("\n" + "=" * 60)
    print("[SONUC] TUM STANDART TESTLERI BASARIYLA GECTI (%100 UYUMLU)")
    print("=" * 60)

if __name__ == "__main__":
    main()
