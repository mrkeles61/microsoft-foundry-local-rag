# Microsoft Foundry Local ve Yerel Yapay Zeka Teknolojileri

## 1. Microsoft Foundry Local Nedir?
Microsoft Foundry Local, geliştiricilerin buluta bağımlı olmadan kendi yerel donanımlarında (Edge / PC) küçük ve orta ölçekli dil modellerini (SLM - Small Language Models) yüksek performansla çalıştırmalarını sağlayan yerel yapay zeka altyapısıdır.

Foundry Local'ın sağladığı temel avantajlar şunlardır:
- **Gizlilik ve Güvenlik:** Hassas kurumsal ve kişisel veriler harici sunuculara veya buluta iletilmez; tüm çıkarım yerel donanımda gerçekleşir.
- **Sıfır Bulut Maliyeti:** API çağrıları veya token bazlı maliyetler olmaksızın sınırsız yerel model kullanımı sunar.
- **Düşük Gecikme Süresi (Zero-Latency Network):** İnternet bağlantısına ihtiyaç duymadan milisaniyeler seviyesinde çıkarım sağlar.

## 2. GPU Optimizasyonu ve CUDA Execution Provider
Microsoft Foundry Local ve ONNX Runtime tabanlı çıkarım motorları varsayılan olarak CPU üzerinde çalışabilir. Ancak NVIDIA CUDA destekli ekran kartlarında maksimum performans elde etmek için iki kritik adım uygulanır:
1. `download_and_register_eps()` fonksiyonu çağrılarak CUDA Execution Provider'lar kayıt edilir.
2. `select_variant()` ile varsayılan CPU yerine GPU varyantı seçilir.

Bu yapılandırma sayesinde RTX serisi ekran kartlarında çıkarım hızında 5x ila 10x arası performans artışı sağlanır.

## 3. Desteklenen Modeller
Foundry Local ekosisteminde özellikle Microsoft Phi-3 (Phi-3-mini, Phi-3-medium), Qwen 2.5 ve Llama-3 serisi modeller optimize edilmiş ONNX ve GGUF formatlarında çalıştırılabilir.
