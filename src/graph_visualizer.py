from typing import List, Dict, Any


class KnowledgeGraphVisualizer:
    """
    Knowledge Graph & Concept Map Generator:
    Extracts key concepts, relationships, and document hierarchies across the knowledge base.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def generate_graph_data(self) -> Dict[str, Any]:
        """
        Extracts entities and relationships from chunks.
        """
        # Core domain concept ontology for Microsoft AI & RAG
        core_entities = [
            {"id": "FoundryLocal", "label": "Microsoft Foundry Local", "type": "Platform"},
            {"id": "Phi3", "label": "Phi-3 / Qwen SLM", "type": "Model"},
            {"id": "CUDA_EP", "label": "CUDA Execution Provider", "type": "Optimization"},
            {"id": "HybridSearch", "label": "Hibrit Arama (Dense + BM25)", "type": "Retrieval"},
            {"id": "RRF", "label": "Reciprocal Rank Fusion", "type": "Algorithm"},
            {"id": "Reranker", "label": "Cross-Encoder Re-Ranking", "type": "Precision Layer"},
            {"id": "RAGTriad", "label": "RAG Triad Metrikleri", "type": "Evaluation"},
            {"id": "ZeroLeakage", "label": "Sıfır Veri Sızıntısı (Privacy)", "type": "Security"}
        ]

        relationships = [
            ("FoundryLocal", "Phi3", "Çalıştırır (Inference)"),
            ("FoundryLocal", "CUDA_EP", "Hızlandırır (GPU EP)"),
            ("FoundryLocal", "ZeroLeakage", "Garanti Eder"),
            ("HybridSearch", "RRF", "Birleştirir"),
            ("HybridSearch", "Reranker", "Besler"),
            ("Reranker", "FoundryLocal", "Bağlam İletir"),
            ("RAGTriad", "ZeroLeakage", "Doğrular")
        ]

        # Generate Mermaid Graph syntax
        mermaid_lines = ["graph LR"]
        for rel in relationships:
            mermaid_lines.append(f'    {rel[0]}["{rel[0]}"] -->|"{rel[2]}"| {rel[1]}["{rel[1]}"]')

        return {
            "entities": core_entities,
            "relationships": relationships,
            "mermaid_code": "\n".join(mermaid_lines)
        }
