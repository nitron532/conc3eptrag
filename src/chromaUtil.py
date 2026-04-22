from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import chromadb
from typing import Any

type chromaDBCollection = chromadb.api.models.Collection.Collection

def createOllamaEmbeddingFunction(modelName: str, url: str):
    return OllamaEmbeddingFunction(
        url = url, #"localhost:11434",
        model_name = modelName # "huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest"
    )

def createOrGetChromaDBCollection(pathToPersistentClient: str, embeddingFunction: chromadb.EmbeddingFunction , collectionName: str):
    client = chromadb.PersistentClient(path = pathToPersistentClient)
    collection = client.get_or_create_collection(name=collectionName, embedding_function = embeddingFunction)
    return collection

def addToCollection(collection: chromaDBCollection, documents: list[str], metadata: dict[str:Any], ids: list[int]):
    collection.add(
        ids=ids,
        documents = documents,
        metadata = metadata,
    )

def queryCollection(collection: chromaDBCollection, queryTexts: list[str], numResults: int):
    return collection.query(
        query_texts=queryTexts,
        n_results=numResults
    )
