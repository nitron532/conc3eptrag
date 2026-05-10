from ollama import Client
from chromadbcollection import PersistentChromaDBCollection
from datasetbuilder import append_training_example
import os
import string
from socket import *
from typing import Any
from itertools import islice
import csv


        #only search materials that are up to the week of the question?
        #include name of file in .cpp files as a comment?
        #have it be able to request more context and mark specific contexts and their ids as bad (ascii table, weird graphics, repeating recursion graphic), add to already searched (might need to be static and then reset on new questions)

        #step 1. first analyze the presented context. are there enough relevant contexts to build a solution from?
        #if not, run re ranker on concept map / query chroma db
        #repeat at most 3 times
        #if the context is still entirely insufficient, state the concepts you think it would require from a cs1 course. 
        #step 2. you are a cs1 student...
        #step 3 classify...
        #step 4 (?). are there any citations/evidence in the answer that contradict your classification?

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def formPrompt(context: list[str], metadatas: list[str], question: str, questionFirst: bool, instruction: str = None):
    prompt = "CONTEXT:"
    for i in range(len(context)):
        documentType = "Handout Page"
        if "Handout" not in metadatas[i]["fileName"]: documentType = "Lecture Page"
        if metadatas[i]["fileType"] != "pdf": documentType = "Code"
        if "HW" in  metadatas[i]["fileName"]: documentType = "Homework"
        prompt += f"\nContext Item {i}:\nFile Name:{metadatas[i]["fileName"]}\nDocument Type:{documentType} \nPage Number:{metadatas[i]["page"]}\nWeek:{metadatas[i]["week"]}\n"
        prompt += f"Chunk Text: {context[i]}"

    if questionFirst: prompt = f"QUESTION: {question}\n" +  prompt
    else: prompt += f"QUESTION: {question}\n"
    if instruction: prompt += instruction

    return prompt

def contactReRanker(similarChunks, question: str, maxReranks: int, alreadySearchedIds: set, filterMetaData: dict[string:Any] = None): #TODO add support for full TCP comms or dump to a file on a shared system
    serverName = '127.0.0.1'
    serverPort = 2020
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))

    toRemove = string.whitespace.replace(' ', '') #TODO verify method 
    table = str.maketrans('', '', toRemove)
    fileName = "contexts.txt"
    negatives = 1

    while(negatives > 0 and maxReranks > 0): #TODO implement persistent connection to avoid tcp overhead and slowstart (but the constant cost is very small at this point)
        with open(f"reranker/{fileName}", 'w+') as f:
            toWrite = ""
            for i, sC in enumerate(similarChunks["documents"][0]):
                toWrite += (f"{similarChunks["ids"][0][i]}:{sC.translate(table)}\n")
                alreadySearchedIds.add(int(similarChunks["ids"][0][i]))
            toWrite += (question.translate(table))
            f.write(toWrite)

        clientSocket.send(f"{fileName}$EOM$".encode())
        serverResponse = clientSocket.recv(1024).decode()
        if serverResponse != f"Server at {serverName}:{serverPort} will rerank {fileName}$EOM$":
            print("Reranking server failed, using first found context items.")
            print("Received from server: ", serverResponse)
            break

        serverResponse = clientSocket.recv(1024).decode() #server reranking, or some error
        rankings = (serverResponse.split(sep = '\n'))[:-1] #remove eom token
        negatives = 0
        for i in range(len(similarChunks["ids"][0])):
            if "-" in rankings[i]:
                negatives += 1
                failedIndex = int(rankings[i][-1])
                similarChunks["ids"][0][failedIndex] = -1
                similarChunks["metadatas"][0][failedIndex] = -1
                similarChunks["documents"][0][failedIndex] = -1 
        if negatives == 0: break

        similarChunks["ids"][0][:] = [x for x in similarChunks["ids"][0] if x != -1]
        similarChunks["metadatas"][0][:] = [x for x in similarChunks["metadatas"][0] if x != -1]
        similarChunks["documents"][0][:] = [x for x in similarChunks["documents"][0] if x != -1]

        maxReranks -= 1
        if maxReranks == 0: break

        mdFilter = {"id":{"$nin": [str(j) for j in alreadySearchedIds]}}

        if filterMetaData:
            mdFilter = {"$and":[filterMetaData, mdFilter]}

        additionalChunks = cs16collection.queryCollection([question], negatives, mdFilter) #with get filtering with metadata, and len()
        for i in range(negatives):
            similarChunks["ids"][0].append(additionalChunks["ids"][0][i])
            alreadySearchedIds.add(int(additionalChunks["ids"][0][i]))
            similarChunks["metadatas"][0].append(additionalChunks["metadatas"][0][i])
            similarChunks["documents"][0].append(additionalChunks["documents"][0][i])
        
            
    clientSocket.close()
    return similarChunks #unnecessary?


qwen = Client(host = "localhost:11434")

cs16collection = PersistentChromaDBCollection("localhost:11434", 
                                              "jinacpu", #"huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest"
                                              "../data/persistent",
                                              "cs16collection")
persistentPath = "../cs16materials"
status = cs16collection.parseAndPopulate(persistentPath) #TODO should have option to just reembed a speciifc document

if status == 0: print(f"Found existing persistent chromadb collection at {persistentPath}")


correctCount = 0 #should have an option for llm to request more context
totalCount = 0

#set up dict for course materials
with open("coursematerials.csv", mode = "r", newline = "") as f:
    reader = csv.DictReader(f)
    materialRows = [row for row in reader]

materialIdsToNames = {}
for rowDict in materialRows:
    materialIdsToNames[int(rowDict["id"])] = rowDict["fileName"]

#set up dict for concepts
with open("concepts.csv", mode = "r", newline = "") as f:
    reader = csv.DictReader(f)
    conceptRows = [row for row in reader]

conceptNamesToMaterialIdLists = {} #depending on how questions are tagged with concepts, you could use conceptIdsToMaterialIds instead (int:int instead of string:int)

for rowDict in conceptRows:
    conceptNamesToMaterialIdLists[rowDict["conceptName"]] = list(map(int,rowDict["materialIds"][1:len(rowDict["materialIds"])-1].split(","))) # conceptName:list[int]


#let llm choose needed topics? but our questions are already labeled with topics

with open("qsfocusans.txt", "r") as f:
    while True:
        linesList = list(islice(f, 3))
        if not linesList: break

        question = linesList[0]
        answer = linesList[1]
        category = linesList[2] #blanket, but keep for now

        materialNames = set()
        totalCount += 1
        alreadySearchedIds = set()

        topicsListEnd = question.find("}")
        topicsList = question[1:topicsListEnd].split() # space separated, with underscores for spaces in names (for parsing). concept map can have spaces for names

        for i,topic in enumerate(topicsList):
            topicsList[i] = topic.replace("_", " ")

        for topic in topicsList:
            for materialId in conceptNamesToMaterialIdLists[topic]:
                materialNames.add(materialIdsToNames[materialId].strip()) #add fileNames to set, avoiding duplicate file names
        
        materialNamesList = list(materialNames)
        mdFilter = {"fileName":{"$in": materialNamesList}}

        question = question[topicsListEnd+1:]

        similarChunksAnswer = cs16collection.queryCollection([answer], 15, mdFilter)
        similarChunksQuestions = cs16collection.queryCollection([question],15, mdFilter)

        similarChunksAnswer = contactReRanker(similarChunksAnswer, answer, 2, alreadySearchedIds, mdFilter)
        similarChunksQuestions = contactReRanker(similarChunksQuestions, question, 2, alreadySearchedIds, mdFilter)

        answerIds = set(similarChunksAnswer["ids"][0])
        questionIdsToIndexes = {}
        for index, id in enumerate(similarChunksQuestions["ids"][0]):
            if id not in answerIds:
                questionIdsToIndexes[id] = index
        
        allSimilarChunks = {"ids":[], "documents":[], "metadatas":[]}
        for id, index in questionIdsToIndexes.items():
            allSimilarChunks["ids"].append(id)
            allSimilarChunks["documents"].append(similarChunksQuestions["documents"][0][index])
            allSimilarChunks["metadatas"].append(similarChunksQuestions["metadatas"][0][index])
            
        for i in range(len(similarChunksAnswer["ids"][0])):
            allSimilarChunks["ids"].append(similarChunksAnswer["ids"][0][i])
            allSimilarChunks["documents"].append(similarChunksAnswer["documents"][0][i])
            allSimilarChunks["metadatas"].append(similarChunksAnswer["metadatas"][0][i])

        print(allSimilarChunks["ids"])

        print(f"Found {len(allSimilarChunks["ids"])} unique related chunks \n")

        toRemove = []

        #rerank with both retrived question and answer chunks to just the question?

            # for concepts, use from concept map?
        analysisMessages = [
            {"role":"system", "content":"""You are an average CS1 student that analyzes CS1 QUESTIONs and ANSWERs.
            You will receive a prompt structured and labeled in this order:
            1. QUESTION: A test question from an introductory CS1 C++ course.
            2. ANSWER: The answer key answer to the QUESTION.
            You will output a response structured like so:
            1. CONCEPTS: A list of CONCEPTs used in the ANSWER to answer the QUESTION.
            2. EXPLANATIONS: For each CONCEPT, a concise analysis of how the ANSWER uses the CONCEPT, limited to one sentence.
            Output only CONCEPTs DIRECTLY found in the ANSWER.
            """}
        ]


        mD = allSimilarChunks["metadatas"]
        sC = allSimilarChunks["documents"]

        firstQuestionPrompt = f"\nQUESTION: {question}\n" + f"\nANSWER:{answer}"

        #TODO replace above prompt formation with the function, for now i need context prompt separated for debugging in the log files



        print(f"\n------QUESTION: {question}\n")
        print(f"\n-------ANSWER: {answer}\n\n")
        analysisMessages.append({"role":"user","content": firstQuestionPrompt})

        response = qwen.chat(
            model = "miniqwenbloom2q8", messages = analysisMessages, think = False
        )

        analysisResponse = response["message"]["content"]

        classifyMessages = [{"role":"user","content": firstQuestionPrompt}]

        classifyMessages.append({"role":"assistant", "content":f"ANALYSIS: {analysisResponse}\n"})

        print("\nANALYSIS RESPONSE -----\n")

        print(analysisResponse)

        contextPrompt = "CONTEXT:"
        for i in range(len(sC)):
            dT = "Handout Page"
            if "Handout" not in mD[i]["fileName"]: dT = "Lecture Page"
            if mD[i]["fileType"] != "pdf": dT = "Code"
            if "HW" in  mD[i]["fileName"]: dT = "Homework"
            contextPrompt += f"\nContext Item {i}:\nFile Name:{mD[i]["fileName"]}\nDocument Type:{dT} \nPage Number:{mD[i]["page"]}\nWeek:{mD[i]["week"]}\n"
            contextPrompt += f"Chunk Text: {sC[i]}"


        # print(f"\n------CONTEXT: {contextPrompt}\n")

        systemPrompt = """
                        You are a CS1 instructor that will label the individual component concepts used in a question under the Revised Bloom's Taxonomy.
                        You will receive an input in this structure:
                        1. QUESTION: A test question from an introductory CS1 C++ course.
                        2. ANSWER: The answer key answer to the QUESTION.
                        3. ANALYSIS: A list consisting of CONCEPTS used in the QUESTION and ANSWER, and EXPLANATIONs of how each CONCEPT was used in the QUESTION and ANSWER.
                        4. CONTEXT: A small subset of CS1 course materials that should be related to the QUESTION. If any CONTEXT items are unhelpful, ignore them.
                        5. INSTRUCTIONS: A guideline you will follow for classifying using the CONTEXT.
                        For each CONCEPT and EXPLANATION together, you will output:
                        1. A Revised Bloom's Taxonomy Level
                        2. A concise explanation of why it falls under this level, limited to one sentence.
                        """

        classifyMessages.insert(0,{"role": "system", "content": systemPrompt}),
        classifyMessages.append({
            "role": "user",
            "content": (
                f"{contextPrompt}\n"
                "INSTRUCTIONS: Using the ANALYSIS above, classify each CONCEPT used in the ANALYSIS with its EXPLANATION "
                "under the Revised Bloom's Taxonomy, citing relevant CONTEXT items only!. "
                "For each CONCEPT, output exactly:\n"
                "CONCEPT: <concept name>\n"
                "LEVEL: <Remember | Understand | Apply | Analyze | Evaluate | Create>\n"
                "REASON: <one sentence citing the specific CONTEXT file that supports this level>\n"
                "Classify every CONCEPT listed in the ANALYSIS. Do not skip any. Here are guidelines for each level:"
                "Remember: If the CONCEPT was recalled from the CONTEXT with no further cognitive load."
                "Understand: If the CONCEPT was explained at a high level in the ANSWER, used to predict output of a code segment, or used to evaluate expressions using ideas from the CONTEXT."
                "Apply: If the CONCEPT was a usage of an algorithm, pattern, procedure or data structure found in the CONTEXT."
                "Analyze: If the CONCEPT was broken down into component parts in the ANSWER, debugged by the ANSWER, or had its purpose explained in the ANSWER." #it keeps analyzing the explanation as analyze lol
                "Evaluate: If the CONCEPT's efficiency was described, was compared against other approaches in terms of efficiency, style, or purpose, or evaluated against a set of criteria."
                "Create: If the CONCEPT was abstracted and combined with other CONCEPTS in a novel way not seen in the CONTEXT, or if a new algorithm resulted from the combination of multiple CONCEPTs."
                "You must cite relevant CONTEXT in your classification, limit your citations to relevant CONTEXT only. If the ANSWER has no code, then it do not label as Apply or Create."
            )
        })

        #add verifier stage to see if a novelty was created? 
        
        response = qwen.chat("miniqwenbloom2q8", messages = classifyMessages, think = False)

        print("\nCLASSES RESPONSE----\n")

        print(response["message"]["content"])