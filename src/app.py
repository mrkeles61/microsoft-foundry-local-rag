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
from src.agents.summarizer import DocumentSummarizer
from src.agents.deep_research import DeepResearchAgent
from src.graph_visualizer import KnowledgeGraphVisualizer

# Page Configuration
st.set_page_config(
    page_title="Microsoft Foundry Local - AI Studio 2.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MICROSOFT FLUENT 2 & GLASSMORPHISM DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient & Modern Typography */
    .hero-container {
        background: linear-gradient(135deg, #0078D4 0%, #004E8C 60%, #002050 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 120, 212, 0.18);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Fluent Cards & Glassmorphism */
    .fluent-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #E1DFDD;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease-in-out;
    }
    .fluent-card:hover {
        border-color: #0078D4;
        box-shadow: 0 6px 22px rgba(0, 120, 212, 0.10);
    }
    
    /* Badges & Chips */
    .agent-chip {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: white;
        margin-right: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    .metric-badge {
        display: inline-flex;
        align-items: center;
        background-color: #F3F2F1;
        border: 1px solid #D2D0CE;
        color: #323130;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    
    /* Quick Pill Buttons */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 120, 212, 0.2);
    }
    
    /* Citation Box */
    .citation-card {
        background-color: #F8F9FA;
        border-left: 4px solid #0078D4;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .citation-highlight {
        background-color: #FFF4CE;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
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
summarizer = DocumentSummarizer(vector_store)
deep_researcher = DeepResearchAgent(vector_store)

# --- SIDEBAR: Settings & Document Management ---
with st.sidebar:
    st.image("https://img.shields.io/badge/Microsoft_AI-Innovators_2026-0078D4?style=for-the-badge&logo=microsoft&logoColor=white", use_container_width=True)
    st.header("⚙️ AI Studio Kontrol Paneli")
    
    with st.expander("Arama & Model Parametreleri", expanded=True):
        search_mode = st.selectbox(
            "Arama Algoritması",
            ["Hibrit (Dense + BM25 + RRF)", "Yalnızca Dense Vektör", "Yalnızca BM25"],
            index=0
        )
        use_reranker = st.checkbox("Cross-Encoder Re-Ranking Aktif", value=True)
        endpoint_url = st.text_input("Local LLM Endpoint", value=DEFAULT_LOCAL_ENDPOINT, help="Foundry Local, Ollama veya LM Studio adresi")
        model_name = st.text_input("Model Adı", value=DEFAULT_MODEL_NAME)
        top_k = st.slider("Top-K Getirilecek Parça", min_value=1, max_value=8, value=FINAL_TOP_K)
        sim_threshold = st.slider("Benzerlik Eşiği", min_value=0.0, max_value=1.0, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05)

    with st.expander("Bölümleme (Chunking) Ayarları", expanded=False):
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)

    st.markdown("---")
    st.header("📁 Doküman Havuzu")
    
    uploaded_files = st.file_uploader(
        "Yeni Dosya Ekle (PDF, TXT, MD, PY, JSON, CSV)",
        type=["pdf", "txt", "md", "py", "json", "csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DOCUMENTS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} yeni belge eklendi.")
        if st.button("🔄 Veritabanını Yeniden İndeksle", type="primary", use_container_width=True):
            with st.spinner("Hibrit indeksleme yapılıyor..."):
                chunks = load_and_chunk_all_documents(DOCUMENTS_DIR, chunk_size, chunk_overlap)
                vector_store.build_index(chunks)
                st.success("Vektör ve BM25 indeksi güncellendi!")
                st.rerun()

    st.subheader("📚 İndekslenmiş Dokümanlar")
    doc_files = [f for f in DOCUMENTS_DIR.glob("*.*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if doc_files:
        for f in doc_files:
            st.markdown(f"- 📄 `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("Doküman bulunamadı.")

    st.markdown("---")
    st.caption(f"Toplam Parça: **{len(vector_store.chunks)}** | Matris: **{vector_store.embeddings.shape}**")


# --- HERO BANNER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Microsoft Foundry Local • AI Studio 2.0</div>
    <div class="hero-subtitle">Çok Ajanlı Niyet Yönlendirici • Hibrit Arama (Dense+BM25) • Re-Ranking • RAG Triad Kalite Ölçümü</div>
</div>
""", unsafe_allow_html=True)


# --- MAIN TABS ---
tab_chat, tab_summary, tab_graph, tab_bench, tab_arch = st.tabs([
    "💬 Multi-Agent RAG Asistanı",
    "📑 Doküman Zekası & Özet",
    "🕸️ İnteraktif Bilgi Grafiği",
    "📊 Canlı Benchmark & Kalite",
    "🏗️ Sistem Mimarisi"
])

rag_engine = RAGEngine(
    vector_store=vector_store,
    endpoint_url=endpoint_url,
    model_name=model_name,
    top_k=top_k,
    similarity_threshold=sim_threshold
)


# --- TAB 1: Multi-Agent Chat ---
with tab_chat:
    # Quick Question Pill Buttons
    st.write("**⚡ Tek Tıkla Örnek Sorular:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    quick_query = None
    with col_q1:
        if st.button("🚀 Foundry Local Nedir?", use_container_width=True):
            quick_query = "Microsoft Foundry Local nedir ve avantajları nelerdir?"
    with col_q2:
        if st.button("⚡ CUDA Hızlandırması", use_container_width=True):
            quick_query = "GPU ve CUDA hızlandırması için hangi fonksiyonlar çağrılmalıdır?"
    with col_q3:
        if st.button("🛡️ Halüsinasyon Koruması", use_container_width=True):
            quick_query = "RAG mimarisinde halüsinasyon nasıl engellenir?"
    with col_q4:
        if st.button("🔧 select_variant Fonksiyonu", use_container_width=True):
            quick_query = "select_variant fonksiyonu ne işe yarar?"

    # Chat History Container
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Merhaba! Ben çok ajanlı RAG 2.0 asistanınızım. Yüklediğiniz dokümanları hem anlamsal (Dense Vektör) hem de anahtar kelime (BM25) ile tarayıp doğrulanmış ve kaynaklı yanıtlar üretiyorum. Nasıl yardımcı olabilirim?",
                "sources": [],
                "evaluation": None,
                "intent_info": None
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show Assigned Agent Badge & Quality Metas
            meta_chips = []
            if msg.get("intent_info"):
                ii = msg["intent_info"]
                meta_chips.append(f'<span class="agent-chip" style="background-color: {ii.get("badge_color", "#0078D4")};">{ii.get("agent_name", "AI")}</span>')

            if msg.get("evaluation"):
                ev = msg["evaluation"]
                meta_chips.append(f'<span class="agent-chip" style="background-color: {ev["badge_color"]};">🛡️ {ev["quality_status"]} (%{ev["confidence_score"]})</span>')
                meta_chips.append(f'<span class="metric-badge">Bağlam: %{ev["context_relevance_pct"]}</span>')
                meta_chips.append(f'<span class="metric-badge">Sadakat: %{ev["groundedness_pct"]}</span>')
                meta_chips.append(f'<span class="metric-badge">Soru Uyumu: %{ev["answer_relevance_pct"]}</span>')

            if meta_chips:
                st.markdown("".join(meta_chips), unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(msg['sources'])} Alıntı)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="citation-card">
                            <b>{idx + 1}. Belge:</b> <code>{src['doc_name']}</code> (Parça #{src['chunk_index']} | Benzerlik: <b>%{src['score']*100:.1f}</b>)<br>
                            <span class="citation-highlight">"{src['text_snippet']}"</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Input handling
    user_input = st.chat_input("Dokümanlarınız hakkında bir soru sorun...")
    active_query = quick_query if quick_query else user_input

    if active_query:
        st.session_state.messages.append({"role": "user", "content": active_query, "sources": [], "evaluation": None, "intent_info": None})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Niyet analiz ediliyor, hibrit arama yapılıyor ve yanıt üretiliyor..."):
                result = rag_engine.query(active_query, history=st.session_state.messages)
                
                st.markdown(result["answer"])
                
                # Render Meta Badges
                meta_chips = []
                ii = result.get("intent_info", {})
                if ii:
                    meta_chips.append(f'<span class="agent-chip" style="background-color: {ii.get("badge_color", "#0078D4")};">{ii.get("agent_name", "AI")}</span>')

                ev = result.get("evaluation", {})
                if ev:
                    meta_chips.append(f'<span class="agent-chip" style="background-color: {ev["badge_color"]};">🛡️ {ev["quality_status"]} (%{ev["confidence_score"]})</span>')
                    meta_chips.append(f'<span class="metric-badge">⚡ Gecikme: {result.get("latency_seconds", 0)}s</span>')

                if meta_chips:
                    st.markdown("".join(meta_chips), unsafe_allow_html=True)

                if result.get("sources"):
                    with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(result['sources'])} Alıntı)", expanded=True):
                        for idx, src in enumerate(result["sources"]):
                            st.markdown(f"""
                            <div class="citation-card">
                                <b>{idx + 1}. Belge:</b> <code>{src['doc_name']}</code> (Parça #{src['chunk_index']} | Benzerlik: <b>%{src['score']*100:.1f}</b>)<br>
                                <span class="citation-highlight">"{src['text_snippet']}"</span>
                            </div>
                            """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "evaluation": result.get("evaluation", None),
            "intent_info": result.get("intent_info", None)
        })

    # Export Report Button
    st.markdown("---")
    col_exp1, col_exp2 = st.columns([2, 3])
    with col_exp1:
        report_lines = ["# 📊 Local RAG 2.0 Araştırma ve Soru-Cevap Raporu\n"]
        for m in st.session_state.messages:
            report_lines.append(f"### {m['role'].upper()}:\n{m['content']}\n")
            if m.get("sources"):
                report_lines.append("**Alıntılanan Kaynaklar:**")
                for s in m["sources"]:
                    report_lines.append(f"- `{s['doc_name']}` (Parça #{s['chunk_index']} | %{s['score']*100:.1f})")
            report_lines.append("\n---\n")
        st.download_button(
            label="📥 Sohbet ve Araştırma Raporunu İndir (.md)",
            data="\n".join(report_lines),
            file_name="RAG_Arastirma_Raporu.md",
            mime="text/markdown",
            use_container_width=True
        )


# --- TAB 2: Document Intelligence & Summarizer ---
with tab_summary:
    st.header("📑 Doküman Zekası & Otomatik Yönetici Özeti")
    st.markdown("Yüklenen dokümanların otomatik özetlerini çıkarın veya iki dokümanı birbiriyle karşılaştırın.")
    
    doc_names = [f.name for f in doc_files]
    if not doc_names:
        st.info("Lütfen sol menüden doküman yükleyin.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("📋 Tekil Doküman Özeti Çıkar")
            selected_doc = st.selectbox("Özetlenecek Dokümanı Seçin", doc_names, key="sum_doc")
            if st.button("✨ Yönetici Özeti Üret", type="primary", use_container_width=True):
                with st.spinner("Doküman taranıyor ve özet çıkarılıyor..."):
                    summary_res = summarizer.summarize_document(selected_doc)
                    st.success(f"**{selected_doc}** Başarıyla Analiz Edildi ({summary_res['chunk_count']} Parça)")
                    
                    st.write("**📌 Ana Başlıklar & Konular:**")
                    for h in summary_res["headings"]:
                        st.markdown(f"- {h}")
                    
                    st.write("**💡 Önemli Noktalar & Vurgular:**")
                    for p in summary_res["key_points"]:
                        st.markdown(f"- {p}")

        with col_s2:
            st.subheader("⚖️ İki Dokümanı Karşılaştır")
            if len(doc_names) >= 2:
                doc_a = st.selectbox("1. Doküman", doc_names, index=0, key="cmp_a")
                doc_b = st.selectbox("2. Doküman", doc_names, index=1, key="cmp_b")
                if st.button("🔄 Karşılaştırmalı Analiz Yap", use_container_width=True):
                    cmp_res = summarizer.compare_documents(doc_a, doc_b)
                    st.table(cmp_res["comparison_table"])
            else:
                st.info("Karşılaştırma yapabilmek için en az 2 doküman yükleyin.")


# --- TAB 3: Knowledge Graph ---
with tab_graph:
    st.header("🕸️ İnteraktif Bilgi Grafiği (Knowledge Graph)")
    st.markdown("Doküman havuzundaki temel kavramların, modellerin ve optimizasyonların ilişki haritası:")
    
    kg_viz = KnowledgeGraphVisualizer(vector_store.chunks)
    graph_data = kg_viz.generate_graph_data()
    
    st.markdown(f"```{graph_data['mermaid_code']}```")
    
    st.subheader("🧩 Varlık ve Kavram Listesi")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("**Kavramlar (Entities):**")
        for ent in graph_data["entities"]:
            st.markdown(f"- **{ent['label']}** (`{ent['type']}`)")
    with col_g2:
        st.write("**İlişkiler (Ontology):**")
        for rel in graph_data["relationships"]:
            st.markdown(f"- `{rel[0]}` ─── *{rel[2]}* ───► `{rel[1]}`")


# --- TAB 4: Live Benchmark Suite ---
with tab_bench:
    st.header("📊 Canlı Kalite & Doğruluk Benchmark Merkezi")
    st.markdown("Sistemin geri getirme (retrieval) kalitesini, gecikme süresini ve RAG Triad skorlarını otomatik test edin.")
    
    if st.button("🚀 Otomatik Benchmark Testini Başlat", type="primary", use_container_width=True):
        test_queries = [
            ("Microsoft Foundry Local nedir?", "Genel Tanım"),
            ("select_variant ve download_and_register_eps ne işe yarar?", "Kod / API"),
            ("RAG mimarisinde halüsinasyon nasıl engellenir?", "Mimari"),
            ("Jupiter gezegeninin uyduları nelerdir?", "Doküman Dışı Güvenlik")
        ]
        
        bench_results = []
        progress_bar = st.progress(0)
        
        for i, (q, category) in enumerate(test_queries):
            start = time.time()
            res = rag_engine.query(q)
            lat = time.time() - start
            ev = res.get("evaluation", {})
            bench_results.append({
                "Soru": q,
                "Kategori": category,
                "Gecikme (sn)": f"{lat:.2f}s",
                "Alıntılanan Parça": len(res["sources"]),
                "Güvenilirlik Skoru": f"%{ev.get('confidence_score', 0)}",
                "Durum": ev.get("quality_status", "Tamamlandı")
            })
            progress_bar.progress((i + 1) / len(test_queries))
            
        st.success("Tüm test senaryoları başarıyla tamamlandı!")
        st.table(bench_results)


# --- TAB 5: Architecture ---
with tab_arch:
    st.header("🏗️ Multi-Agent RAG Ultra Mimarisi")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Dokümanlar: PDF, TXT, MD, PY, JSON, CSV] --> B[Smart Sentence Splitter & Loader]
        B --> C[Metin Parçaları - 500 Char / 80 Overlap]
        C --> D1[Dense Embeddings: SentenceTransformers]
        C --> D2[Sparse Index: BM25 Okapi Index]
        
        Q[Kullanıcı Sorusu] --> R[Intent Router Ajanı]
        R -->|Teknik| AG1[🛠️ Teknik Ajan]
        R -->|Özet| AG2[📋 Özet Ajanı]
        R -->|Derin| AG3[🔬 Derin Araştırma Ajanı]
        R -->|Genel| AG4[💡 Bilgi Ajanı]
        
        AG1 --> S[Hibrit Arama: Dense + BM25 + RRF]
        AG2 --> S
        AG3 --> S
        AG4 --> S
        
        S --> RERANK[Cross-Encoder Re-Ranking Katmanı]
        RERANK --> TOP[En Yüksek Hassasiyetli Top-K Parçalar]
        TOP --> PROMPT[Grounded Prompt Context + Halüsinasyon Koruması]
        PROMPT --> LLM[Yerel SLM: Microsoft Foundry Local / Phi-3 / Qwen]
        LLM --> OUT[Doğrulanmış Cevap + Kaynak Alıntıları + RAG Triad Metrikleri]
    ```
    """)
