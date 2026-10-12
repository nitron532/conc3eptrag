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

This project was created for the constructive alignment of UCSB's CS16 course, so a lot of the code is written to interface with other teaching team backend scripts.

It has only been tested on Ubuntu 22.04.

To start, run `pip install -r requirements.txt` in a conda or venv.
You'll also need to install [Ollama](https://docs.ollama.com/linux), and PostgreSQL if you don't have already.

For inputting questions, you should have a JSON containing a single list, each object representing a question and containing at least the following keys:
```
content - string
id - int
tags - list[string]
```
**content** is a string containing a question and answer written in LaTex. It should have AnswerArea, item, and Solution environments/commands for info to be parsed correctly.

**id** is a unique integer identifier for each question.

**tags** is a list of concepts. These are used to focus the RAG retrieval on only course materials tagged with those concepts, providing the classification more relevant context.

Run `py latexparse.py <input_json_path> <output_json_path>` to get a JSON containing parsed question/answer info ready to feed into Conc3ept.




