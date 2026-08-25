import streamlit as st
import os
import time
from pathlib import Path
from src.config import (
    DOCUMENTS_DIR,
    VECTOR_STORE_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_LOCAL_ENDPOINT,
    DEFAULT_MODEL_NAME
)
from src.loader import load_and_chunk_all_documents, extract_text_from_file
from src.vector_store import LocalVectorStore
from src.engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Microsoft Foundry Local - RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0078D4;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #f3f6f9;
        border-left: 4px solid #0078D4;
        padding: 10px 15px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #E1DFDD;
        color: #323130;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    if VECTOR_STORE_PATH.exists():
        store.load(VECTOR_STORE_PATH)
    else:
        # Initial indexing on first launch
        chunks = load_and_chunk_all_documents(DOCUMENTS_DIR)
        store.build_index(chunks)
    return store


vector_store = get_vector_store()

# --- SIDEBAR: Settings & Document Management ---
with st.sidebar:
    st.image("https://img.shields.io/badge/Microsoft_AI-Innovators_2026-0078D4?style=for-the-badge&logo=microsoft&logoColor=white", use_container_width=True)
    st.header("⚙️ RAG Yapılandırması")
    
    with st.expander("Model & Endpoint Ayarları", expanded=False):
        endpoint_url = st.text_input("Local LLM Endpoint", value=DEFAULT_LOCAL_ENDPOINT, help="Foundry Local, Ollama veya LM Studio endpoint adresi")
        model_name = st.text_input("Model Adı", value=DEFAULT_MODEL_NAME)
        top_k = st.slider("Top-K Getirilecek Parça Sayısı", min_value=1, max_value=8, value=DEFAULT_TOP_K)
        sim_threshold = st.slider("Benzerlik Eşiği (Threshold)", min_value=0.0, max_value=1.0, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05)

    with st.expander("Bölümleme (Chunking) Ayarları", expanded=False):
        chunk_size = st.number_input("Chunk Size (Karakter)", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk Overlap (Karakter)", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)

    st.markdown("---")
    st.header("📁 Doküman Yönetimi")
    
    # File Uploader
    uploaded_files = st.file_uploader("Yeni Belge Yükle (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DOCUMENTS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} yeni belge kaydedildi.")
        if st.button("🔄 Veritabanını Yeniden İndeksle", type="primary"):
            with st.spinner("Belgeler indeksleniyor..."):
                chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size, chunk_overlap)
                vector_store.build_index(chunks)
                st.success("Vektör veritabanı başarıyla güncellendi!")
                st.rerun()

    # List Indexed Documents
    st.subheader("📚 Mevcut Dokümanlar")
    doc_files = list(DOCUMENTS_DIR.glob("*.*"))
    if doc_files:
        for f in doc_files:
            st.markdown(f"- 📄 `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("Henüz doküman yüklenmedi.")

    st.markdown("---")
    st.caption(f"Toplam İndekslenen Parça: **{len(vector_store.chunks)}**")


# --- MAIN CONTENT TABS ---
tab_chat, tab_docs, tab_arch = st.tabs(["💬 RAG Sohbet & Soru-Cevap", "📑 İndeks Denetleyicisi", "🏗️ Sistem Mimarisi"])

# Initialize RAG Engine
rag_engine = RAGEngine(
    vector_store=vector_store,
    endpoint_url=endpoint_url,
    model_name=model_name,
    top_k=top_k,
    similarity_threshold=sim_threshold
)

# --- TAB 1: Chat Interface ---
with tab_chat:
    st.markdown('<div class="main-title">🤖 Local RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Microsoft Foundry Local Altyapısı ile Yerel, Güvenli ve Doğrulanabilir Doküman Asistanı</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Merhaba! Ben yerel RAG asistanınızım. Yüklediğiniz dokümanlara dayalı olarak sorularınızı cevaplayabilir ve doğrudan kaynak gösterebilirim. Size nasıl yardımcı olabilirim?",
                "sources": []
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(msg['sources'])} Alıntı)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**{idx + 1}. Belge:** `{src['doc_name']}` (Parça #{src['chunk_index']} | Benzerlik Skoru: **%{src['score']*100:.1f}**)\n\n"
                            f"> *\"{src['text_snippet']}\"*"
                        )

    # Chat Input
    if user_query := st.chat_input("Dokümanlarınız hakkında bir soru sorun... (Örn: Microsoft Foundry Local nedir?)"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Yerel vektör tabanında taranıyor ve yanıt üretiliyor..."):
                start_time = time.time()
                result = rag_engine.query(user_query)
                latency = time.time() - start_time
                
                st.markdown(result["answer"])
                st.caption(f"⚡ Yanıt Süresi: {latency:.2f} sn | 📚 Bulunan Parça: {len(result['sources'])}")

                if result.get("sources"):
                    with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(result['sources'])} Alıntı)", expanded=True):
                        for idx, src in enumerate(result["sources"]):
                            st.markdown(
                                f"**{idx + 1}. Belge:** `{src['doc_name']}` (Parça #{src['chunk_index']} | Benzerlik Skoru: **%{src['score']*100:.1f}**)\n\n"
                                f"> *\"{src['text_snippet']}\"*"
                            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", [])
        })


# --- TAB 2: Document & Chunk Inspector ---
with tab_docs:
    st.header("📑 İndekslenmiş Metin Parçaları (Chunks)")
    if len(vector_store.chunks) == 0:
        st.info("Henüz indekslenmiş bir parça bulunmuyor. Sidebar'dan belgelerinizi yükleyin.")
    else:
        st.write(f"Vektör veritabanında toplam **{len(vector_store.chunks)}** parça bulunmaktadır.")
        
        search_filter = st.text_input("Parçalar içinde kelime ara:", "")
        filtered_chunks = [c for c in vector_store.chunks if search_filter.lower() in c["text"].lower()] if search_filter else vector_store.chunks
        
        for c in filtered_chunks[:20]:
            with st.expander(f"📄 {c['doc_name']} - Parça #{c['chunk_index']} ({len(c['text'])} Karakter)"):
                st.code(c["text"], language="markdown")


# --- TAB 3: Architecture & Info ---
with tab_arch:
    st.header("🏗️ Sistem Mimarisi ve RAG Çalışma Mantığı")
    st.markdown("""
    Bu uygulama **Microsoft AI Innovators / Summer School 2026** kapsamında geliştirilmiş olup, 
    tamamen yerel ortamda çalışan ve veri gizliliğini koruyan bir RAG (Retrieval-Augmented Generation) mimarisidir.
    """)
    
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Kullanıcı Dokümanları (PDF/TXT/MD)] --> B[Document Loader & Sentence Splitter]
        B --> C[Metin Parçaları - Chunks (500 Karakter)]
        C --> D[Yerel Embedding Modeli - SentenceTransformers]
        D --> E[Vektör İndeksi - Cosine Distance Matrisi]
        
        F[Kullanıcı Sorusu] --> G[Soru Vektörleştirme]
        G --> H[Vektör Arama - Top-K Cosine Retrieval]
        E --> H
        H --> I[En Alakalı Parçalar & Kaynaklar]
        I --> J[Prompt Context Birleştirici + Halüsinasyon Koruması]
        J --> K[Yerel Dil Modeli - Foundry Local / Phi-3 / Qwen]
        K --> L[Doğrulanmış Yanıt + Kaynak Alıntıları]
    ```
    """)
    
    st.markdown("""
    ### 🔑 Temel Özellikler:
    1. **Sıfır Veri Sızıntısı (Zero Data Leakage):** Dokümanlar ve sorular yerel cihaz dışına çıkmaz.
    2. **Halüsinasyon Engelleme:** Dokümanda yer almayan bilgi için model kesinlikle uydurma yapmaz.
    3. **Açık Kaynak Referanslama:** Her yanıtta kullanılan belgenin adı, parça numarası ve benzerlik oranı şeffafça sunulur.
    4. **Esnek LLM Desteği:** Microsoft Foundry Local, LM Studio veya Ollama ile tak-çalıştır entegrasyon.
    """)
