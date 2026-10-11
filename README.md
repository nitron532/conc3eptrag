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
- Boost.ASIO (network library, using TLS functionality for distributed processing)
- Unix Domain Sockets (for single-machine hosting)
  
UI Stack:
- React (frontend)
- Flask (backend)
- PostgreSQL (metadata storage)

*conc3ept UI lives in a separate repository

Given a json of questions and course materials, Conc3ept's end-to-end pipeline streams these into the classification system, parses into ChromaDB, and context-searches with a course concept map to focus RAG context. Reranked course materials and questions are given to a prompt chain classification process and explanations of classifications with citations to course material streamed into an output json, ready to use in the UI.
