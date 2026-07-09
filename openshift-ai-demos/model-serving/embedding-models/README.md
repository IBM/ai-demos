# Embedding Models

Embedding models convert text into dense numerical vectors (embeddings) that capture semantic meaning. These embeddings enable applications such as Retrieval-Augmented Generation (RAG), semantic search, and vector databases by allowing text to be compared based on meaning rather than exact keywords.

This demo shows how to deploy and serve embedding models on Red Hat OpenShift AI using vLLM. It provides an example deployment for serving embedding models and generating vector embeddings through a scalable and efficient inference service.

---

## Available Runtimes

### vLLM Runtime
High-performance inference engine supporting embedding models via the pooling runner.

**Available Models:**
- Granite Embedding 125M English
- BGE-M3

[→ View vLLM deployment guide](vllm/)
