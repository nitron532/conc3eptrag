# Conc3ept
### A self-hosted distributed RAG classification system for constructive alignment of assessments.

Conc3ept is a free, open-source AI-powered tool that helps instructors evaluate whether their assessments align with their course materials using the Revised Bloom's Taxonomy's cognitive levels, and concept mapping.

More technically, it is a self-hosted, distributed RAG (retrieval-augmented generation) question classification pipeline that retrieves context from course materials and an instructor-defined course concept map to give multi-label Revised Bloom's Taxonomy classifications to questions. It is designed with data privacy and practicality in mind, aiding instructors in evaluating the constructive alignment of their assessments and course materials without worrying about nosey third-parties or resource constraints.

#### Inputs
- A JSON of your course's questions
- Course materials such as lectures and code examples
- An instructor-defined concept map of their course's concepts
  
#### Outputs
- LLM-identified concepts, mapped back to an instructor-defined map of course concepts
- For each concept:
  - A Revised Bloom's Taxonomy cognitive level
  - A cited explanation for the classification, referencing relevant course materials

These outputs help instructors check what their assessments are testing: 

**Are students simply remembering facts from lecture, or using creative problem solving to tackle unseen problems?**

## System Diagrams

| Course Material Parsing and Embedding | Context Retrieval and Reranking |
| :---: | :---: |
| ![Course material parsing and embedding process. Chunks are stored in ChromaDB for retrieval and reranking.](images/parseandembed.png) | ![Context retrieval and reranking process. A question/answer pair is inputted and relevant course context chunks are outputted to the classification process.](images/retrieveandrerank.png) |


This project was created for the constructive alignment of UCSB's CS16 course, so a lot of the code is written to interface with other teaching team backend scripts.

It has only been tested on Ubuntu 22.04.

**This project is still under development, mainly focusing on user experience as most functionality is implemented.**
## Stack
RAG Stack:
- Ollama (local classification LLM and embedding model hosting)
- llama.cpp (self-hosted re-ranker server specifically implemented for jina-reranker-v3 architecture)
- ChromaDB (vector database for parsed course materials, connects to Ollama for embedding model)
- safetensors (C++ API for copy-free tensor manipulation, used for jina-reranker-v3 tensor operations)
  
Parsing stack:
- PyMuPDF (PDF text layer extraction)
- PaddleOCR (OCR, only when images are detected)
  
Network / IPC Stack:
- Boost.ASIO (network library, using TLS for fast and secure distributed processing)
- Unix Domain Sockets (for single-machine hosting, fast and secure)
  
UI Stack*:
- React (frontend)
- Flask (backend)
- PostgreSQL (metadata storage)

*conc3ept UI lives in a [separate repository](https://github.com/nitron532/conc3ept)

## Install
To start, run 

`pip install -r requirements.txt` 

preferably in a conda or venv.

You'll also need to install [Ollama](https://docs.ollama.com/linux) and [PostgreSQL](https://www.postgresql.org/download/linux/ubuntu/) if you haven't already.

You may need to redirect Ollama's environment variables to point to a model folder of your choosing.

For the reranker server, it is specifically implemented for jina-reranker-v3, and looks for a .gguf file (can be found [here](https://huggingface.co/jinaai/jina-reranker-v3-GGUF)).
  
For ChromaDB and classification, you can pull any model using Ollama (make sure its pointed to the correct directory).
You may also want to create custom model files to finetune performance and restrict some models to CPU or GPU.
The tested implementation puts jina-code-embeddings-1.5b on CPU, and a Qwen3.5 9B on GPU.

### CMake Build
cd into `src/reranker` and run:
```
mkdir build
cd build
cmake -S .. -B.
cmake --build . -j
```
This will build llama.cpp automatically and include the header-only Boost.asio and safetensors libraries.

### Reranker and SSL
If you plan on using TCP/TLS, you'll need to set up > TLS 1.2.

[mkcert](https://github.com/filosottile/mkcert) is an easy tool to set up SSL certificates for development, which should be enough if you keep the CA Root and server's private key private.

Before starting the main process below, make sure to build the reranker server with CMake and start it via

`./rerankerserver <1 for tcp, or 0 for uds> ` 

then, depending on your IPC method add:
```
- TCP/TLS
  - --cert <path to server cert> --key <path to server private key>
- UDS
  - --sockPath <socket path that will be created>
```

Also make sure you have the (jina-reranker-v3.gguf file([https://huggingface.co/jinaai/jina-reranker-v3-GGUF] in the models folder at the top level directory. 

## Usage / Implementation Description
Before starting Conc3ept, make sure you start Ollama and the reranker server so that embedding and reranking are online and ready to go.

### Question Input
For inputting questions, you should have a JSON containing a single list, each object representing a question and containing at least the following keys:
```
content: string
id: int
tags: list[string]
```
**content** is a string containing a question and answer written in LaTex. It should have AnswerArea, item, and Solution environments/commands for info to be parsed correctly.

**id** is a unique integer identifier for each question.

**tags** is a list of concepts. These are used to focus the RAG retrieval on only course materials tagged with those concepts, providing the classification more relevant context.

Run 

`py latexparse.py <input_json_path> <output_json_path>`

to get a JSON containing parsed question/answer info ready to feed into Conc3ept.

### Course Material Parsing
***TODO: generalize from oldrepo and improve path passing***

Currently a path must be written in the code to a directory containing your course materials. All materials should be at the top level of this dedicated directory.

Running 

`py contactLLM.py <1 to map oldrepo tags to concept map> <1 for TCP/TLS , 0 for UDS>`

will automatically parse and embed the course materials into ChromaDB if they are unparsed (marked by PostgreSQL), or go straight to retrieval and classification if all are parsed.

The path to the outputted parsed questions must also be written in code.

The default implementation supports `.cpp`, `.py`, and `.cc` files, but this can be easily extended by adding onto the conditional's list in line 77 of `parsing.py`:

`elif os.path.splitext(file)[1] in [".cpp", ".py", ".cc", ".h", ".html", ".java", ...]`

PyMuPDF will be used to parse any PDFs, and if any images are detected, PaddleOCR is used parse text. Code files are treated as plaintext and stored as they are.

Parsing yields parsed text via the generator `walkParse`, allowing memory efficient and checkpointed parsing. Yielded text is added to a persistent ChromaDB collection.

Parsed files are marked "parsed" in the PostgreSQL database, and document ids persist across parse runs so the generator picks up where it left off if it was interrupted.

### Course Material Retrieval and Reranking
***TODO docs for assigning concept tags to materials in postgres***

After parsing is finished and the course materials are embedded into ChromaDB, questions are streamed from the input question JSON one at a time as a question/answer pair.

Course materials tagged with the same concepts as the question/answer pair serve as the corpus for retrieval, and the most cosine-similar documents are retrieved from ChromaDB, and passed to the reranker server for further processing. The first retrieval is a quick, large retrieval, whereas the reranking cross encodes the query with each document to create a query and doc embedding, which are then compared with cosine similarity to produce a final similarity score. The difference here is that the embeddings used in the first retrieval were created independently, whereas the reranker embeds query and document with respect to eachother.

If the reranker finds too many unrelated documents, these unrelated documents will be excluded and the concept search breadth is expanded by one layer in the concept map. This repeats until a user-defined number of rounds, or until enough relevant documents are found.

### Prompt Chaining Classification
The surviving documents from the retrieval/rerank stage are passed as context along with the question/answer pair to the classification LLM.

The prompt chain consists of the following stages, separated to encourage unbiased outputs:

| Stage | Input | Output | Description |
| -------- | -------- | -------- | -------- |
| Analysis | Question/Answer pair | List of LLM-identified concepts | This stage identifies concepts independent of course material and concept map to ensure unbiased identification. |
| Classification | Question/Answer pair, retrieved course materials, and the Analysis' concept list | Revised Bloom's Taxonomy cognitive level for each concept, with a short explanation citing the provided course materials. | This stage provides the main output of the tool that is used for constructive alignment evaluation. |
| Mapping | Analysis' concept list | Mappings from Analysis concepts to instructor-defined course concepts | This stage maps the LLM-identified concepts back to the instructor's course concept map for organization. It does not depend on the output of Classification. |

When a question/answer pair finishes this prompt chain process, it is streamed out into an output JSON, again for memory efficiency and checkpointing.

## Next Steps
Besides the TODOs, I would like to
- Explore finetuning the classifier with DSPy, and maybe the embedder and reranker in general
- Explore using Jev ai from Typesafe (but it's not free)
- Support different reranking models
- Isolate the prompt-chaining part of the system, the heaviest, so that functionality can be loaded onto an HPC server
