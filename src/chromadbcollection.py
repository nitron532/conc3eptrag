from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import chromadb
from typing import Any
import parsing as parsing

class PersistentChromaDBCollection:
    def __init__(self, 
                 ollamaURL: str,
                 modelName: str,
                 pathToPersistentClient: str,
                 collectionName: str
                 ):
        self.embeddingFunction = OllamaEmbeddingFunction( #will extend to support others. but this is for local llms
            url = ollamaURL,
            model_name = modelName
        )
        client = chromadb.PersistentClient(path = pathToPersistentClient)
        self.collection = client.get_or_create_collection(name = collectionName, embedding_function = self.embeddingFunction)
        self.startId = 0

    def addToCollection(self,
                        documents: list[str],
                        metadata: dict[str:Any],
                        ids: list[int]):
            try:
                self.collection.add(
                    ids = ids,
                    documents = documents,
                    metadatas = metadata
                )
            except Exception as e:
                print(e)
                raise #some error. should be more descriptive in the future
    
    def parseAndPopulate(self,
                             materialsPath: str):
        if self.startId != 0: return 0 #nothing done, collection was initialized already.
        try:
            for chunks, ids, metadata, latestId in parsing.walkParse(materialsPath, self.startId):
                self.addToCollection(chunks, metadata, ids)
                self.startId = latestId
        except Exception as e:
            print(e)
            raise #some error. should be more descriptive in the future
        return 1 #success

    def queryCollection(self,
                        queryTexts: list[str],
                        numResults: int):
        return self.collection.query(
            query_texts=queryTexts,
            n_results=numResults 
        )

    def getCollectionCount(self): return self.collection.count()

    def collectionGet(self, metadata:dict[str:Any]):
        return self.collection.get(
            where=metadata
        )
