import os
import chromaUtil as chromaUtil
import parsing as parsing

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# persistentPath = "../data/persistent"
# materials = "../cs16materials"
# collectionName = "cs16db"

def getMaterialDB(persistentPath: str, materials: str, collectionName: str):
    #create jina code embeddings to embed handouts and slides for chroma db
    ollamaEmbedder = chromaUtil.createOllamaEmbeddingFunction("huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest","localhost:11434")

    # persistent chroma db
    collection = chromaUtil.createOrGetChromaDBCollection(persistentPath, ollamaEmbedder, collectionName)

    if collection.count() == 0: #TODO allow adding new data to an existing collection. also add data type metadata (image->ocr, or just text), sort ocrd images into weeks for dev?
        print(f"Did not find a persistent chroma database at {persistentPath} with data. Creating/getting persistent db and populating from {materials} folder now.")
        for chunks, ids, metadata in parsing.walkParse(materials):
            collection.add(
                ids = ids,
                documents = chunks,
                metadatas = metadata
            )

    return collection


if __name__ == "__main__":
    collection = getMaterialDB("../data/persistent", "../cs16materials", "cs16db")
    print("Input a query, or 'exit' to exit the program:")
    q = input()
    while q != "exit":
        result = collection.query(
            query_texts=[q],
            n_results=4
        )
        print("\n-----------------METADATA v-------------------\n")
        print(result["metadatas"])
        print("\n---------METADATA ^----DOCUMENTS v------------\n")
        print(result["documents"])
        print("\n-----------------DOCUMENTS ^-------------------\n")
        print("Input a query, or 'exit' to exit the program.")
        q = input()