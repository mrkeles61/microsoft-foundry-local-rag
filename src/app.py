import streamlit as st
import os
import time
import json
from pathlib import Path
from src.config import (
    DOCUMENTS_DIR,
    VECTOR_STORE_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    FINAL_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_LOCAL_ENDPOINT,
    DEFAULT_MODEL_NAME,
    SUPPORTED_EXTENSIONS
)
from src.loader import load_and_chunk_all_documents
from src.vector_store import LocalVectorStore
from src.engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Microsoft Foundry Local - RAG 2.0 Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0078D4;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #605E5C;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #EDEBE9;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    store = LocalVectorStore(store_path=VECTOR_STORE_PATH)
    if VECTOR_STORE_PATH.exists():
        store.load(VECTOR_STORE_PATH)
    else:
        chunks = load_and_chunk_all_documents(DOCUMENTS_DIR)
        store.build_index(chunks)
    return store


vector_store = get_vector_store()

# --- SIDEBAR: Settings & Document Management ---
with st.sidebar:
    st.image("https://img.shields.io/badge/Microsoft_AI-Innovators_2026-0078D4?style=for-the-badge&logo=microsoft&logoColor=white", use_container_width=True)
    st.header("⚙️ RAG 2.0 Yapılandırması")
    
    with st.expander("Arama & Model Ayarları", expanded=True):
        search_mode = st.selectbox(
            "Arama Stratejisi",
            ["Hibrit (Dense Vector + BM25 + RRF)", "Yalnızca Yoğun Vektör (Dense)", "Yalnızca Seyrek Kelime (BM25)"],
            index=0
        )
        use_reranker = st.checkbox("Cross-Encoder Re-Ranking Aktif", value=True, help="Aday parçaları hassas sıralama katmanından geçirir.")
        endpoint_url = st.text_input("Local LLM Endpoint", value=DEFAULT_LOCAL_ENDPOINT, help="Foundry Local, Ollama veya LM Studio adresi")
        model_name = st.text_input("Model Adı", value=DEFAULT_MODEL_NAME)
        top_k = st.slider("Top-K Getirilecek Parça Sayısı", min_value=1, max_value=8, value=FINAL_TOP_K)
        sim_threshold = st.slider("Benzerlik Eşiği (Threshold)", min_value=0.0, max_value=1.0, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05)

    with st.expander("Bölümleme (Chunking) Parametreleri", expanded=False):
        chunk_size = st.number_input("Chunk Size (Karakter)", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk Overlap (Karakter)", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)

    st.markdown("---")
    st.header("📁 Doküman Yönetimi")
    
    uploaded_files = st.file_uploader(
        "Belge Yükle (PDF, TXT, MD, PY, JSON, CSV)",
        type=["pdf", "txt", "md", "py", "json", "csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DOCUMENTS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} yeni belge kaydedildi.")
        if st.button("🔄 Veritabanını Yeniden İndeksle", type="primary"):
            with st.spinner("Hibrit indeksleme yapılıyor..."):
                chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size, chunk_overlap)
                vector_store.build_index(chunks)
                st.success("Vektör ve BM25 indeksi başarıyla güncellendi!")
                st.rerun()

    st.subheader("📚 İndekslenmiş Dokümanlar")
    doc_files = [f for f in DOCUMENTS_DIR.glob("*.*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if doc_files:
        for f in doc_files:
            st.markdown(f"- 📄 `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("Henüz doküman yüklenmedi.")

    st.markdown("---")
    st.caption(f"Toplam Parça: **{len(vector_store.chunks)}** | Yoğun Matris: **{vector_store.embeddings.shape}**")


# --- MAIN CONTENT TABS ---
tab_chat, tab_metrics, tab_docs, tab_arch = st.tabs([
    "💬 RAG 2.0 Sohbet & Asistan",
    "📊 Canlı RAG Triad Analitiği",
    "📑 İndeks Denetleyicisi",
    "🏗️ Sistem Mimarisi (RAG 2.0)"
])

# Initialize Engine
rag_engine = RAGEngine(
    vector_store=vector_store,
    endpoint_url=endpoint_url,
    model_name=model_name,
    top_k=top_k,
    similarity_threshold=sim_threshold
)

# --- TAB 1: Chat Interface ---
with tab_chat:
    st.markdown('<div class="main-title">⚡ Local RAG 2.0 with Microsoft Foundry Local</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hibrit Arama (Dense+BM25) • Re-Ranking • Halüsinasyon Koruması • Çok Turlu Hafıza</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Merhaba! Ben RAG 2.0 asistanınızım. Yüklediğiniz dokümanları hem anlamsal (vektör) hem de anahtar kelime (BM25) ile tarayarak doğrulanmış yanıtlar üretiyorum. Size nasıl yardımcı olabilirim?",
                "sources": [],
                "evaluation": None
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show Quality Evaluation Badge if present
            if msg.get("evaluation"):
                ev = msg["evaluation"]
                st.markdown(
                    f'<span class="badge-chip" style="background-color: {ev["badge_color"]};">'
                    f'🛡️ {ev["quality_status"]} (%{ev["confidence_score"]})</span> '
                    f'<small>Bağlam: %{ev["context_relevance_pct"]} | Doğruluk: %{ev["groundedness_pct"]} | Soru Uyumu: %{ev["answer_relevance_pct"]}</small>',
                    unsafe_allow_html=True
                )

            # Show Sources Expander
            if msg.get("sources"):
                with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(msg['sources'])} Alıntı)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**{idx + 1}. Belge:** `{src['doc_name']}` (Parça #{src['chunk_index']} | Benzerlik Skoru: **%{src['score']*100:.1f}**)\n\n"
                            f"> *\"{src['text_snippet']}\"*"
                        )

    if user_query := st.chat_input("Dokümanlarınız hakkında soru sorun... (Örn: select_variant fonksiyonu ne işe yarar?)"):
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": [], "evaluation": None})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Hibrit arama yapılıyor, parçalar yeniden sıralanıyor ve yanıt üretiliyor..."):
                result = rag_engine.query(user_query, history=st.session_state.messages)
                
                st.markdown(result["answer"])
                
                # Quality Badge
                ev = result.get("evaluation", {})
                if ev:
                    st.markdown(
                        f'<span class="badge-chip" style="background-color: {ev["badge_color"]};">'
                        f'🛡️ {ev["quality_status"]} (%{ev["confidence_score"]})</span> '
                        f'<small>Gecikme: {result.get("latency_seconds", 0)} sn | Parça: {len(result["sources"])}</small>',
                        unsafe_allow_html=True
                    )

                if result.get("sources"):
                    with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(result['sources'])} Alıntı)", expanded=True):
                        for idx, src in enumerate(result["sources"]):
                            st.markdown(
                                f"**{idx + 1}. Belge:** `{src['doc_name']}` (Parça #{src['chunk_index']} | Benzerlik: **%{src['score']*100:.1f}**)\n\n"
                                f"> *\"{src['text_snippet']}\"*"
                            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "evaluation": result.get("evaluation", None)
        })

    # Export Chat Report Button
    st.markdown("---")
    col_exp1, col_exp2 = st.columns([1, 4])
    with col_exp1:
        if st.button("📥 Raporu İndir (.md)"):
            report_lines = ["# 📊 Local RAG 2.0 Araştırma ve Soru-Cevap Raporu\n"]
            for m in st.session_state.messages:
                report_lines.append(f"### {m['role'].upper()}:\n{m['content']}\n")
                if m.get("sources"):
                    report_lines.append("**Alıntılanan Kaynaklar:**")
                    for s in m["sources"]:
                        report_lines.append(f"- `{s['doc_name']}` (Parça #{s['chunk_index']} | %{s['score']*100:.1f})")
                report_lines.append("\n---\n")
            report_content = "\n".join(report_lines)
            st.download_button(
                label="Dosyayı Kaydet",
                data=report_content,
                file_name="RAG_Arastirma_Raporu.md",
                mime="text/markdown"
            )


# --- TAB 2: RAG Triad Analytics ---
with tab_metrics:
    st.header("📊 Canlı RAG Triad ve Kalite Ölçümleri")
    st.markdown("""
    RAG Triad, modern kurumsal yapay zeka sistemlerinde doğruluğu ölçen 3 temel metriktir:
    1. **Context Relevance:** Getirilen belgelerin soruyla uyumu.
    2. **Groundedness:** Cevabın belgelere sadakati (Halüsinasyon olmama oranı).
    3. **Answer Relevance:** Cevabın kullanıcının asıl sorusunu yanıtlama derecesi.
    """)

    last_assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant" and m.get("evaluation")]
    if last_assistant_msgs:
        last_eval = last_assistant_msgs[-1]["evaluation"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🛡️ Genel Güvenilirlik", f"%{last_eval['confidence_score']}")
        with col2:
            st.metric("🎯 Context Relevance", f"%{last_eval['context_relevance_pct']}")
        with col3:
            st.metric("🔒 Groundedness (Sadakat)", f"%{last_eval['groundedness_pct']}")
        with col4:
            st.metric("💡 Answer Relevance", f"%{last_eval['answer_relevance_pct']}")
    else:
        st.info("Metrikleri görmek için sohbet sekmesinden soru sorabilirsiniz.")


# --- TAB 3: Document Inspector ---
with tab_docs:
    st.header("📑 İndekslenmiş Metin Parçaları")
    if len(vector_store.chunks) == 0:
        st.info("Henüz indekslenmiş parça bulunmuyor.")
    else:
        st.write(f"Toplam **{len(vector_store.chunks)}** parça taranabilir durumda.")
        search_filter = st.text_input("Parçalarda kelime ara:", "")
        filtered = [c for c in vector_store.chunks if search_filter.lower() in c["text"].lower()] if search_filter else vector_store.chunks
        for c in filtered[:25]:
            with st.expander(f"📄 {c['doc_name']} - Parça #{c['chunk_index']} ({len(c['text'])} Karakter)"):
                st.code(c["text"], language="markdown")


# --- TAB 4: Architecture & Workflow ---
with tab_arch:
    st.header("🏗️ RAG 2.0 İleri Düzey Sistem Mimarisi")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Dokümanlar: PDF, TXT, MD, PY, JSON, CSV] --> B[Smart Sentence Splitter & Loader]
        B --> C[Metin Parçaları - 500 Char / 80 Overlap]
        C --> D1[Dense Embeddings: SentenceTransformers]
        C --> D2[Sparse Index: BM25 Okapi Index]
        
        Q[Kullanıcı Sorusu] --> Q1[Çok Turlu Hafıza & Query Reformulation]
        Q1 --> S1[Dense Semantic Search]
        Q1 --> S2[Sparse Keyword BM25 Search]
        
        D1 --> S1
        D2 --> S2
        
        S1 --> RRF[Reciprocal Rank Fusion - RRF]
        S2 --> RRF
        
        RRF --> RERANK[Cross-Encoder Re-Ranking Katmanı]
        RERANK --> TOP[En Yüksek Hassasiyetli Top-K Parçalar]
        TOP --> PROMPT[Grounded Prompt Context + Halüsinasyon Koruması]
        PROMPT --> LLM[Yerel SLM: Microsoft Foundry Local / Phi-3 / Qwen]
        LLM --> OUT[Doğrulanmış Cevap + Kaynak Alıntıları + RAG Triad Metrikleri]
    ```
    """)
