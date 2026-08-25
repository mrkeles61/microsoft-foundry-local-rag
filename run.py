import sys
import os
import argparse
from pathlib import Path

# Enable UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DOCUMENTS_DIR, VECTOR_STORE_PATH
from src.loader import load_and_chunk_all_documents
from src.vector_store import LocalVectorStore
from src.engine import RAGEngine


def run_cli():
    print("=" * 60)
    print("Local RAG with Foundry Local - Terminal Modu")
    print("=" * 60)
    
    # Initialize vector store
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    if not VECTOR_STORE_PATH.exists():
        print(f"[CLI] Dokümanlar taranıyor: {DOCUMENTS_DIR}")
        chunks = load_and_chunk_all_documents(DOCUMENTS_DIR)
        store.build_index(chunks)
    else:
        store.load(VECTOR_STORE_PATH)
        
    engine = RAGEngine(vector_store=store)
    print("\n[OK] RAG Motoru hazir! Cikmak icin 'q' veya 'exit' yazabilirsiniz.\n")

    while True:
        try:
            query = input("\nSorunuz: ").strip()
            if not query:
                continue
            if query.lower() in ["q", "exit", "quit"]:
                print("Cikiliyor...")
                break

            print("\nYanit araniyor...\n")
            result = engine.query(query)
            
            print("--- CEVAP ---")
            print(result["answer"])
            
            if result.get("sources"):
                print("\n--- KAYNAKLAR ---")
                for i, src in enumerate(result["sources"]):
                    print(f"[{i+1}] Belge: {src['doc_name']} (Parca #{src['chunk_index']} | Benzerlik: %{src['score']*100:.1f})")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nCikis yapildi.")
            break


def run_web():
    app_path = BASE_DIR / "src" / "app.py"
    cmd = f'streamlit run "{app_path}"'
    print(f"[LAUNCH] Streamlit arayuzu baslatiliyor: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Application Launcher")
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode instead of web UI")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_web()
