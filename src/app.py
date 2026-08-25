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
    page_title="Microsoft Foundry Local - Multi-Agent RAG Ultra",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0078D4;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #605E5C;
        margin-bottom: 1.2rem;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        color: white;
    }
    .quick-pill {
        display: inline-block;
        background-color: #F3F2F1;
        border: 1px solid #D2D0CE;
        padding: 6px 12px;
        border-radius: 16px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        cursor: pointer;
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
    st.header("⚙️ Çok Ajanlı RAG Ayarları")
    
    with st.expander("Model & Arama Ayarları", expanded=True):
        search_mode = st.selectbox(
            "Arama Yöntemi",
            ["Hibrit (Dense + BM25 + RRF)", "Yalnızca Dense Vektör", "Yalnızca BM25"],
            index=0
        )
        use_reranker = st.checkbox("Cross-Encoder Re-Ranking", value=True)
        endpoint_url = st.text_input("Local LLM Endpoint", value=DEFAULT_LOCAL_ENDPOINT)
        model_name = st.text_input("Model Adı", value=DEFAULT_MODEL_NAME)
        top_k = st.slider("Top-K Parça Sayısı", min_value=1, max_value=8, value=FINAL_TOP_K)
        sim_threshold = st.slider("Benzerlik Eşiği", min_value=0.0, max_value=1.0, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05)

    with st.expander("Bölümleme Parametreleri", expanded=False):
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)

    st.markdown("---")
    st.header("📁 Doküman Havuzu")
    
    uploaded_files = st.file_uploader(
        "Yeni Dosya Yükle (PDF, TXT, MD, PY, JSON, CSV)",
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
                st.success("Vektör veritabanı başarıyla güncellendi!")
                st.rerun()

    st.subheader("📚 Mevcut Dokümanlar")
    doc_files = [f for f in DOCUMENTS_DIR.glob("*.*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if doc_files:
        for f in doc_files:
            st.markdown(f"- 📄 `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("Doküman bulunamadı.")

    st.markdown("---")
    st.caption(f"Toplam Parça: **{len(vector_store.chunks)}**")


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
    st.markdown('<div class="main-title">⚡ Multi-Agent Local RAG Ultra</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Microsoft Foundry Local • Niyet Yönlendirici • Hibrit Arama • RAG Triad Metrikleri</div>', unsafe_allow_html=True)

    # Quick Question Pills
    st.write("**⚡ Tek Tıkla Örnek Sorular:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    quick_query = None
    with col_q1:
        if st.button("🚀 Foundry Local Nedir?"):
            quick_query = "Microsoft Foundry Local nedir ve avantajları nelerdir?"
    with col_q2:
        if st.button("⚡ CUDA Hızlandırması"):
            quick_query = "GPU ve CUDA hızlandırması için hangi fonksiyonlar çağrılmalıdır?"
    with col_q3:
        if st.button("🛡️ Halüsinasyon Koruması"):
            quick_query = "RAG mimarisinde halüsinasyon nasıl engellenir?"
    with col_q4:
        if st.button("🔧 select_variant Fonksiyonu"):
            quick_query = "select_variant fonksiyonu ne işe yarar?"

    # Chat Messages History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Merhaba! Ben çok ajanlı RAG 2.0 asistanınızım. Yüklediğiniz dokümanları hem semantik hem de anahtar kelime tabanlı tarayıp doğrulanmış ve kaynaklı yanıtlar sunuyorum. Bir soru sorarak başlayabilirsiniz.",
                "sources": [],
                "evaluation": None,
                "intent_info": None
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show Assigned Agent Badge
            if msg.get("intent_info"):
                ii = msg["intent_info"]
                st.markdown(
                    f'<span class="badge-chip" style="background-color: {ii.get("badge_color", "#0078D4")};">'
                    f'{ii.get("agent_name", "AI Asistanı")}</span>',
                    unsafe_allow_html=True
                )

            # Show Quality Evaluation Badge
            if msg.get("evaluation"):
                ev = msg["evaluation"]
                st.markdown(
                    f'<span class="badge-chip" style="background-color: {ev["badge_color"]};">'
                    f'🛡️ {ev["quality_status"]} (%{ev["confidence_score"]})</span> '
                    f'<small>Bağlam: %{ev["context_relevance_pct"]} | Sadakat: %{ev["groundedness_pct"]} | Soru Uyumu: %{ev["answer_relevance_pct"]}</small>',
                    unsafe_allow_html=True
                )

            if msg.get("sources"):
                with st.expander(f"🔍 Doğrulanmış Kaynaklar ({len(msg['sources'])} Alıntı)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**{idx + 1}. Belge:** `{src['doc_name']}` (Parça #{src['chunk_index']} | Benzerlik Skoru: **%{src['score']*100:.1f}**)\n\n"
                            f"> *\"{src['text_snippet']}\"*"
                        )

    # Input handling (either from chat input or quick pill button)
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
                
                # Intent & Quality Badges
                ii = result.get("intent_info", {})
                if ii:
                    st.markdown(
                        f'<span class="badge-chip" style="background-color: {ii.get("badge_color", "#0078D4")};">'
                        f'{ii.get("agent_name", "AI Asistanı")}</span>',
                        unsafe_allow_html=True
                    )

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
            "evaluation": result.get("evaluation", None),
            "intent_info": result.get("intent_info", None)
        })

    # Download Report Button
    st.markdown("---")
    col_exp1, col_exp2 = st.columns([1, 4])
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
            label="📥 Sohbet Raporunu İndir (.md)",
            data="\n".join(report_lines),
            file_name="RAG_Arastirma_Raporu.md",
            mime="text/markdown"
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
            if st.button("✨ Yönetici Özeti Üret", type="primary"):
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
                if st.button("🔄 Karşılaştırmalı Analiz Yap"):
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
    
    if st.button("🚀 Otomatik Benchmark Testini Başlat", type="primary"):
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
