# Conc3ept
### A self-hosted distributed RAG classification system for constructive alignment of assessments.
RAG Stack:
- Ollama (local classification LLM hosting)
- llama.cpp (self-hosted re-ranker server for custom jina architecture)
- ChromaDB (vector database for parsed course materials)
Parsing stack:
- PyMuPDF (PDF text layer extraction)
- PaddleOCR (OCR when images are detected)
Network / IPC Stack:
- OpenSSL (secure distributed communication)
- Boost.ASIO (network library)
UI Stack:
- React (frontend)
- Flask (backend)
- PostgreSQL (metadata storage)

*conc3ept UI lives in a separate repository
