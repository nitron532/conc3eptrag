from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import chromadb
from typing import Any
import parsing
import psycopg
from paddleocr import PaddleOCR

class PersistentChromaDBCollection:
    def __init__(self, 
                 ollamaURL: str,
                 modelName: str,
                 pathToPersistentClient: str,
                 collectionName: str,
                 space: str,
                 psqldb: str
                 ):
        self.embeddingFunction = OllamaEmbeddingFunction( #will extend to support others. but this is for local llms
            url = ollamaURL,
            model_name = modelName
        )
        client = chromadb.PersistentClient(path = pathToPersistentClient)
        #cosine for jina
        self.collection = client.get_or_create_collection(name = collectionName, embedding_function = self.embeddingFunction, configuration={"hnsw":{"space":space}})
        self.startId = 0

        #check if there is anything in the db. if so, set the start id to the highest id + 1
        results = self.collection.get()
        if results["ids"]:
            self.startId = int(results["ids"][-1]) + 1 #plus one so the next chunk gets the next id
        
        self.conn = psycopg.connect( #TODO may need to modify for user perm psql instead of superuser
            host = "localhost",
            dbname = psqldb,
        )
        print(f"Successfuly connected to both ChromaDB collection {pathToPersistentClient} and PostgreSQL database {psqldb}")

    def addToCollection(self,
                        documents: list[str],
                        metadata: dict[str:Any],
                        ids: list[int],
                        embeddings: list[list[float]]) -> None: 
            try:
                self.collection.add(
                    #embeddings = [...]    #TODO change to add the embedding itself returned by the embedding server
                    ids = ids,
                    documents = documents,
                    metadatas = metadata
                )
            except Exception as e:
                print(e)
                raise #some error. TODO should be more descriptive in the future
    
    def parseAndPopulate(self,
                             materialsPath: str) -> int:
        if self.startId != 0: 
            # a persistent client exists. check for any new materials, or any unparsed ones. materials list should be updated from web interface
            updated = True
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT \"fileName\" from coursematerials WHERE parsed = false OR parsed IS NULL "
                )
                ocr = None
                unparsedFiles = []
                for res in cur.fetchall():
                    if ".pdf" in res: ocr = PaddleOCR(lang = 'en', use_angle_cls = True, device = "gpu") #can change to cpu if needed
                    unparsedFiles.append(res[0])
                if len(unparsedFiles) < 1: return 0
                for fileName in unparsedFiles:
                    chunks, ids, metadata, latestId, parsed = parsing.formFileChunks(materialsPath, fileName, ocr, 300, self.startId)
                    if parsed:
                        updated = True
                        self.addToCollection(chunks, metadata, ids)
                        self.startId = latestId
                        cur.execute(
                            "UPDATE coursematerials SET parsed = %s WHERE \"fileName\" = %s",
                            (parsed, fileName)
                        )
                        print(f"Updated collection with {fileName}")
            if updated:
                return 1 # collection updated successfuly

            return 0 #nothing done, collection was initialized already, unparsed files were empty or some error occurred in parsing that resulted in no change.
        try:
            for chunks, ids, metadata, latestId, file in parsing.walkParse(materialsPath, self.startId):
                if latestId != -1:
                    self.addToCollection(chunks, metadata, ids)
                    self.startId = latestId
                parsed = True if latestId != -1 else False
                if parsed:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "UPDATE coursematerials SET parsed = %s WHERE \"fileName\" = %s",
                            (parsed, file)
                        )
                        self.conn.commit()
                    
        except Exception as e:
            print(e)
            raise #some error. TODO should be more descriptive in the future
        return 1 #success

    def queryCollection(self,
                        queryTexts: list[str],
                        numResults: int,
                        filterMetaData: dict[str:Any] = None):
        return self.collection.query(
            query_texts=queryTexts,
            n_results=numResults,
            where = filterMetaData
        )

    def getCollectionCount(self) -> int: return self.collection.count()

    def getCollectionStartId(self) -> int: return self.startId

    def collectionGet(self, metadata:dict[str:Any]):
        return self.collection.get(
            where=metadata
        )
