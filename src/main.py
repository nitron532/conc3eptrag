import os
import chromaUtil as chromaUtil
import parsing as parsing

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#create jina code embeddings to embed handouts and slides for chroma db
ollamaEmbedder = chromaUtil.createOllamaEmbeddingFunction("huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest","localhost:11434")

# persistent chroma db
persistentPath = "../data/persistent"
materials = "../cs16materials"
collectionName = "cs16db"
collection = chromaUtil.createOrGetChromaDBCollection(persistentPath, ollamaEmbedder, collectionName)

if collection.count() == 0:
    print(f"Did not find a persistent chroma database at {persistentPath} with data. Creating/getting persistent db and populating from {materials} folder now.")
    chunks, ids, metadata = parsing.walkParse(materials)
    collection.add(
        ids = ids,
        documents = chunks,
        metadatas = metadata
    )

result = collection.query(
    query_texts=["Describe two ways to alter a 2d array."],
    n_results=4
)

print(result["metadatas"])
print()
print(result["documents"])