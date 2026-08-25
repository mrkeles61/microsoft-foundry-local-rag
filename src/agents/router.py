import re
from enum import Enum
from typing import Dict, Any, Tuple


class QueryIntent(Enum):
    TECHNICAL_CODE = "technical_code"
    EXECUTIVE_SUMMARY = "executive_summary"
    DEEP_RESEARCH = "deep_research"
    FACTUAL_QA = "factual_qa"


class IntentRouter:
    """
    Multi-Agent Intent Classification Router:
    Analyzes user queries to route them to the specialized agent persona and adjusts
    retrieval depth, system prompts, and formatting.
    """
    def __init__(self):
        self.code_keywords = [
            "kod", "code", "fonksiyon", "function", "cuda", "onnx", "gpu", "sdk", "api",
            "select_variant", "download_and_register_eps", "python", "script", "kütüphane",
            "parametre", "endpoint", "install", "kurulum"
        ]
        self.summary_keywords = [
            "özet", "genel", "nedir", "kısaca", "summary", "overview",
            "madde", "avantaj", "fayda", "benefit", "temel", "nelerdir"
        ]
        self.research_keywords = [
            "karşılaştır", "fark", "analiz", "derin", "compare",
            "vs", "difference", "detaylı", "mimari", "değerlendir"
        ]

    def _matches_any(self, text: str, keywords: list) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def route(self, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """
        Classifies query intent using robust substring/stem matching.
        """
        # 1. Deep Research
        if self._matches_any(query, self.research_keywords):
            return QueryIntent.DEEP_RESEARCH, {
                "agent_name": "🔬 Derin Araştırma Ajanı (Deep Research)",
                "agent_icon": "🔬",
                "top_k_multiplier": 2,
                "persona_prompt": "Sen kapsamlı teknik analizler ve çok boyutlu karşılaştırmalar yapan bir Kıdemli AI Araştırmacısısın.",
                "badge_color": "#8764B8"
            }

        # 2. Technical Code
        if self._matches_any(query, self.code_keywords):
            return QueryIntent.TECHNICAL_CODE, {
                "agent_name": "🛠️ Teknik & Mimari Ajanı (System Architect)",
                "agent_icon": "🛠️",
                "top_k_multiplier": 1,
                "persona_prompt": "Sen Microsoft sistemleri, SDK'lar ve yerel model optimizasyonunda uzmanlaşmış bir Sistem Mimarı ve Kıdemli Yazılım Mühendisisin.",
                "badge_color": "#0078D4"
            }

        # 3. Executive Summary
        if self._matches_any(query, self.summary_keywords):
            return QueryIntent.EXECUTIVE_SUMMARY, {
                "agent_name": "📋 Yönetici Özeti Ajanı (Executive Brief)",
                "agent_icon": "📋",
                "top_k_multiplier": 1,
                "persona_prompt": "Sen karmaşık teknik raporları yöneticiler için net, maddeli ve özlü özetlere dönüştüren bir Çözüm Danışmanısın.",
                "badge_color": "#107C41"
            }

        # 4. Default Factual QA
        return QueryIntent.FACTUAL_QA, {
            "agent_name": "💡 Doğrulanmış Bilgi Ajanı (Grounded QA)",
            "agent_icon": "💡",
            "top_k_multiplier": 1,
            "persona_prompt": "Sen yerel dokümanlara dayalı doğrulanmış ve kaynaklı bilgiler sunan bir RAG Asistanısın.",
            "badge_color": "#008272"
        }
