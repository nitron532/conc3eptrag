import os
import string
from socket import *
from typing import Any
# from itertools import islice
import ijson
import csv
import re
import json
from graph import addNeighboringConcepts
import sys
from datetime import datetime

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

oldRepoTags = int(sys.argv[1]) #1 if yes, 0 if directly mapped to concept map
#find concepts : 1 if you want llm to identify concepts and map back to map for material search

endpoint = None
portNum = None
comms = None
clientSocket = None
while(comms != "UDS" and comms != "TCP"):
    comms = input("(UDS) or (TCP) to connect to reranker? ")
    try:
        if comms == "TCP":
            endpoint = input("IP?")
            portNum = input("Port?")
            clientSocket = socket(AF_INET, SOCK_STREAM)
            endpoint = (endpoint, int(portNum))
        elif comms == "UDS":
            endpoint = "/tmp/conc3ept"
            clientSocket = socket(AF_UNIX, SOCK_STREAM)
    except Exception as e:
        print(f"Error: {e}")
clientSocket.connect(endpoint)

#TODO could just have this spawn the reranker server as a child process so the user doesnt have to set up that server either

from ollama import Client
from chromadbcollection import PersistentChromaDBCollection

def formPrompt(context: list[str], metadatas: list[str], question: str, questionFirst: bool, instruction: str = None):
    prompt = "CONTEXT:"
    for i in range(len(context)):
        documentType = "Handout Page"
        if "Handout" not in metadatas[i]["fileName"]: documentType = "Lecture Page"
        if metadatas[i]["fileType"] != "pdf": documentType = "Code"
        if "HW" in  metadatas[i]["fileName"]: documentType = "Homework"
        prompt += f"\nContext Item {i}:\nFile Name:{metadatas[i]["fileName"]}\nDocument Type:{documentType} \nPage Number:{metadatas[i]["page"]}\n" #\nWeek:{metadatas[i]["week"]}
        prompt += f"Chunk Text: {context[i]}"

    if questionFirst: prompt = f"QUESTION: {question}\n" +  prompt
    else: prompt += f"QUESTION: {question}\n"
    if instruction: prompt += instruction

    return prompt


def contactReRanker(similarChunks, query: str, maxReranks: int, 
                    alreadySearchedIds: set, clientSocket: socket,
                    conceptIds: list[int], edgeRows: list[dict],
                    conceptNamesToMaterialIdLists: dict[string:list[int]],
                    materialNames: set[str],
                    filterMetaData: dict[string:Any] = None): 

    toRemove = string.whitespace.replace(' ', '') #TODO verify method 
    table = str.maketrans('', '', toRemove)
    negatives = 1

    while(negatives > 0 and maxReranks > 0):
        ogLength = len(similarChunks["documents"][0])
        toWrite = ""
        for i, sC in enumerate(similarChunks["documents"][0]):
            toWrite += (f"{similarChunks["ids"][0][i]}:{sC.translate(table)}\n")
            alreadySearchedIds.add(int(similarChunks["ids"][0][i]))
        toWrite += (query.translate(table))

        clientSocket.send(f"{toWrite}$EOM$".encode())

        serverResponse = ""
        while("$EOM$" not in serverResponse):
            serverResponse += clientSocket.recv(1024).decode() #server reranking, or some error
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
            #add to filterMetadata neighboring concept materialNames ($in materialnames)
            #gradually expand out every rerank sort of like BFS
            if negatives >= ogLength // 1.5: #threshold
                conceptsList = addNeighboringConcepts(conceptIds, edgeRows, conceptIdsToConceptNames)
                for topic in conceptsList:
                    for materialId in conceptNamesToMaterialIdLists[topic]:
                        materialNames.add(materialIdsToNames[materialId].strip())
                print("bfs'd concept list:", conceptsList)

            #materials from returned updated topic list, one BFS level out
            filterMetaData["fileName"]["$in"] = list(materialNames)
            mdFilter = {"$and":[filterMetaData, mdFilter]}

        additionalChunks = cs16collection.queryCollection([query], negatives, mdFilter) #with get filtering with metadata, and len()
        #if len(additional) < negatives: bfs?
        for i in range(len(additionalChunks["ids"][0])):
            similarChunks["ids"][0].append(additionalChunks["ids"][0][i])
            alreadySearchedIds.add(int(additionalChunks["ids"][0][i]))
            similarChunks["metadatas"][0].append(additionalChunks["metadatas"][0][i])
            similarChunks["documents"][0].append(additionalChunks["documents"][0][i])
        
    return similarChunks #unnecessary?


def parseCSVIntoDict(path: str):
    with open(path, mode = "r", newline = "") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows

ollama = Client(host = "localhost:11434")

cs16collection = PersistentChromaDBCollection("localhost:11434", 
                                              "jinacpu", #"huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest"
                                              "../data/persistent",
                                              "cs16collection",
                                              "cosine",
                                              "conc3ept")
persistentPath = "../cs16materials"
status = cs16collection.parseAndPopulate(persistentPath) 
#TODO should have option to just reembed a speciifc document. same file names will stay the same in psql even if updated
#from web interface remove materials if you removed materials from the directory
#TODO change csv parsing to psql queries

if status == 0: print(f"Found existing persistent chromadb collection at {persistentPath}")

#set up dict for course materials
materialRows = parseCSVIntoDict("coursematerials.csv")

materialIdsToNames = {}
for rowDict in materialRows:
    materialIdsToNames[int(rowDict["id"])] = rowDict["fileName"]

#set up dict for concepts
conceptRows = parseCSVIntoDict("concepts.csv")

#parse edges
edgeRows = parseCSVIntoDict("edges.csv")

#are all these maps really needed
conceptNamesToMaterialIdLists = {} #depending on how questions are tagged with concepts, you could use conceptIdsToMaterialIds instead (int:int instead of string:int)
conceptNamesToConceptIds = {}
conceptIdsToConceptNames = {}
for rowDict in conceptRows:
    conceptNamesToMaterialIdLists[rowDict["conceptName"]] = list(map(int,rowDict["materialIds"][1:len(rowDict["materialIds"])-1].split(","))) # conceptName:list[int]
    conceptNamesToConceptIds[rowDict["conceptName"]] = int(rowDict["id"])
    conceptIdsToConceptNames[int(rowDict["id"])] = rowDict["conceptName"]

oldRepoTagToCM = {
    "Coding": "Functions",
    "Variable": "Variables",
    "Argv": "Command Line Arguments",
    "Loop": "Loops",
    "LinkedList": "Linked List",
    "DataStructure": "Arrays",
    "Numbers": "Math",
    "Recursive": "Recursion",
    "Boolean": "Booleans",
    "Inheritance": "Classes",
    "MemoryAddress": "Memory Addresses",
    "Search": "Arrays",
    "Vector": "Vectors",
    "Cstring": "C String",
    "Array": "Arrays",
    "Shfiting": "Math",
    "Git": "Command Line Arguments", #not exactly... but ok
    "Alias": "Memory Addresses",
    "Heap": "Memory Layout",
    "access specifier": "Variables",
    "constructor": "Classes",
    "Multi-dim Array": "Arrays",
    "Pass by reference": "Call By Reference/Value",
    "Iterative": "Loops",
    "Branches": "Conditionals",
    "Operator Overloading": "Classes",
    "typedef": "Variables",
    "Memory Model": "Memory Models",
    "namespace": "Namespaces",
    "Command Line": "Command Line Arguments",
    "Overflow": "Math",
    "Type Casting": "Variables",
    "strings": "Strings",
    "const": "Variables"
}

returnId = 0

questionFile = "parsedcs16questionswithexam.json" 
# questionFile = "oldrepotest.json"
resultFile = "oldreporesults3.jsonl" 
# resultFile = "ordebug.jsonl"
logFile = "oldreporesults3.txt"
# logFile = "ordebug.txt"
errorFile = "oldrepoerrors3.txt"
retryThreshold = 3
with open(questionFile, "r") as f:
    questionList = ijson.items(f, "item")
    for questionObject in questionList:
        retries = 0
        success = False
        while retries < retryThreshold and not success:
            try:
                question = questionObject["question"]
                answer = questionObject["answer"]
                qId = questionObject["id"]
                conceptsList = questionObject["tags"]

                if qId < returnId: break #oldrepo ids are exported in increasing order
                print("oldrepo question id", qId)


                if oldRepoTags:
                    for i in range(len(conceptsList)):
                        try:
                            conceptsList[i] = oldRepoTagToCM[conceptsList[i]]
                        except Exception as e: #no entry in the dict, so the name is the same as the CM
                            pass
                    conceptsList = list(set(conceptsList)) #remove duplicates
                print("cm concept list:", conceptsList)
                    
                materialNames = set()
                alreadySearchedIds = set()

                #add fileNames to set, avoiding duplicate file names
                for topic in conceptsList:
                    for materialId in conceptNamesToMaterialIdLists[topic]:
                        materialNames.add(materialIdsToNames[materialId].strip())

                if not conceptsList: #no concepts found? search over all context. later, call LLM to do this
                    for fileName in materialIdsToNames.values():
                        materialNames.add(fileName.strip())
                
                mdFilter = {"fileName":{"$in": list(materialNames)}}

                query = question + answer if answer is not None else question

                similarChunks = cs16collection.queryCollection(queryTexts = [query], filterMetaData = mdFilter, numResults = 20)
                #Remove text that was added to set embedder inference mode from retrieved chunks
                for i in range(len(similarChunks)):
                    similarChunks["documents"][0][i] = similarChunks["documents"][0][i][similarChunks["documents"][0][i].find('\n')+1:]

                similarChunks = contactReRanker(similarChunks = similarChunks, query = query, maxReranks = 2,
                                                        alreadySearchedIds = alreadySearchedIds,
                                                        clientSocket = clientSocket,
                                                        conceptIds = [conceptNamesToConceptIds[name] for name in conceptsList],
                                                        edgeRows = edgeRows,
                                                        conceptNamesToMaterialIdLists = conceptNamesToMaterialIdLists,
                                                        materialNames = materialNames,
                                                        filterMetaData = mdFilter)

                print(similarChunks["ids"])

                print(f"Found {len(similarChunks["ids"][0])} unique related chunks \n")

                analysisMessages = [
                    {"role":"system", "content":"""You are an average CS1 student that analyzes CS1 QUESTIONs and ANSWERs.
                    You will receive a prompt structured and labeled in this order:
                    1. QUESTION: A test question from an introductory CS1 C++ course.
                    2. ANSWER: The answer key answer to the QUESTION.
                    You will output a response structured with these capitalized headers:
                    1. CONCEPTS: An enumerated list of CONCEPTs used in the ANSWER to answer the QUESTION.
                    2. EXPLANATIONS: An list of explanations for each CONCEPT labeled by the CONCEPT, each a concise analysis of how the ANSWER uses the CONCEPT, limited to one sentence.
                    Output only CONCEPTs DIRECTLY found in the ANSWER.
                    """}
                ]

                mD = similarChunks["metadatas"][0]
                sC = similarChunks["documents"][0]

                firstQuestionPrompt = f"\nQUESTION: {question}\n" + f"\nANSWER:{answer}"

                #TODO replace above prompt formation with the function, for now i need context prompt separated for debugging in the log files

                with open(logFile, "a") as r:
                    r.write(f"---------------------{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}-----------------------------------------------------------------------------------------------------------------------------------------------------")
                    r.write(f"\n------QUESTION: {question}\n")
                    r.write(f"\n-------ANSWER: {answer}\n\n")

                analysisMessages.append({"role":"user","content": firstQuestionPrompt})

                response = ollama.chat(
                    model = "miniqwenbloom2q8", messages = analysisMessages, think = False
                )

                analysisResponse = response["message"]["content"]

                classifyMessages = [{"role":"user","content": firstQuestionPrompt}]

                classifyMessages.append({"role":"assistant", "content":f"ANALYSIS: {analysisResponse}\n"})

                with open(logFile, "a") as r:
                    r.write(f"\nANALYSIS RESPONSE ---{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}--\n{analysisResponse}")

                contextPrompt = "CONTEXT:"
                for i in range(len(sC)):
                    dT = "Handout Page"
                    if "Handout" not in mD[i]["fileName"]: dT = "Lecture Page"
                    if mD[i]["fileType"] != "pdf": dT = "Code"
                    if "HW" in  mD[i]["fileName"]: dT = "Homework"
                    contextPrompt += f"\nContext Item {i}:\nFile Name:{mD[i]["fileName"]}\nDocument Type:{dT} \nPage Number:{mD[i]["page"]}\n" #Week:{mD[i]["week"]}\n
                    contextPrompt += f"Chunk Text: {sC[i]}"

                systemPrompt = """
                                You are a CS1 instructor that will label the individual component concepts used in a question under the Revised Bloom's Taxonomy.
                                You will receive an input in this structure:
                                1. QUESTION: A test question from an introductory CS1 C++ course.
                                2. ANSWER: The answer-key answer to the QUESTION.
                                3. ANALYSIS: A list consisting of CONCEPTS used in the QUESTION and ANSWER, and EXPLANATIONs of how each CONCEPT was used in the QUESTION and ANSWER.
                                4. CONTEXT: A small subset of CS1 course materials that should be related to the QUESTION. If any CONTEXT items are unhelpful, ignore them.
                                5. INSTRUCTIONS: A guideline you will follow for classifying using the CONTEXT.
                                For each CONCEPT and EXPLANATION together, you will output in the same order as the ANALYSIS:
                                1. A Revised Bloom's Taxonomy Level
                                2. A concise explanation of why it falls under this level, limited to one sentence.
                                """

                classifyMessages.insert(0,{"role": "system", "content": systemPrompt}),
                classifyMessages.append({
                    "role": "user",
                    "content": (
                        f"{contextPrompt}\n"
                        "INSTRUCTIONS: Using the ANALYSIS above, classify each CONCEPT used in the ANALYSIS with its EXPLANATION "
                        "under the Revised Bloom's Taxonomy, citing relevant CONTEXT items only!."
                        "For each CONCEPT, output in the same order as the ANALYSIS exactly:\n"
                        "CONCEPT: <concept name>\n"
                        "LEVEL: <Remember | Understand | Apply | Analyze | Evaluate | Create>\n"
                        "REASON: <one sentence citing the specific CONTEXT items, by citing them each individually in double dollar signs (e.g. $$1$$ $$2$$) that supports the LEVEL>\n"
                        "Classify every CONCEPT listed in the ANALYSIS. Do not skip any. Here are guidelines for each level:"
                        "Remember: If the CONCEPT was recalled from the CONTEXT with no further cognitive load.\n"
                        "Understand: If the CONCEPT was explained at a high level in the ANSWER, used to predict output of a code segment, or used to evaluate expressions using ideas from the CONTEXT.\n"
                        "Apply: If the CONCEPT was a usage of an algorithm, pattern, procedure or data structure found in the CONTEXT.\n"
                        "Analyze: If the CONCEPT was broken down into component parts in the ANSWER, debugged by the ANSWER, or had its purpose explained in the ANSWER.\n" #it keeps analyzing the explanation as analyze lol
                        "Evaluate: If the CONCEPT's efficiency was described, was compared against other approaches in terms of efficiency, style, or purpose, or evaluated against a set of criteria.\n"
                        "Create: If the CONCEPT was abstracted and combined with other CONCEPTS in a novel way not seen in the CONTEXT, or if a new algorithm resulted from the combination of multiple CONCEPTs.\n"
                        "You will cite relevant CONTEXT in your classification, limit your citations to relevant CONTEXT only. If the ANSWER has no code, then it do not label as Apply or Create."
                    )
                })

                response = ollama.chat("miniqwenbloom2q8", messages = classifyMessages, think = False)

                classificationResponse = response["message"]["content"]

                with open(logFile, "a") as r:
                    r.write(f"\nCLASSES RESPONSE- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}---\n{classificationResponse}")
                    r.write("\n------------------ENDCLASSES------------------------------\n")
                    
                llmConcepts = analysisResponse[analysisResponse.find("EXPLANATIONS:\n") + 14:]
                expectedConceptCount = classificationResponse.count("CONCEPT:")

                conceptMessages = [{"role":"user", "content":f"""
                    Given this list of CS1 Concepts:
                    {list(conceptNamesToMaterialIdLists.keys())}

                    Map EVERY LLMCONCEPT below to exactly one CS1 Concept from the list above.
                    Output EXACTLY one mapping per line, with NO blank lines between them.
                    Format STRICTLY as: <LLM CONCEPT NAME>:<CS1 Concept>
                    DO NOT combine multiple mappings on one line.
                    DO NOT add any explanation or extra text.
                    DO NOT skip any concept.

                    LLMCONCEPTS:
                    {llmConcepts}
                    """}
                ]

                response = ollama.chat("miniqwenbloom2q8", messages = conceptMessages, think = False)

                conceptMapConcepts = []
                conceptResponse = response["message"]["content"]
                with open(logFile,"a") as r:
                    r.write(f"\nID'DCONCEPTSBYLLM --{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}--: {conceptResponse}\n")
                colon = 0
                newline = 0
                print("conceptResponse:", conceptResponse)

                conceptMapConcepts = []
                for line in conceptResponse.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    colonIndex = line.find(":")
                    if colonIndex == -1:
                        continue
                    mapped = line[colonIndex + 1:].strip()
                    # validate it's actually in the concept map
                    if mapped in conceptNamesToMaterialIdLists:
                        conceptMapConcepts.append(mapped)
                    else:
                        retries += 1
                        if retries < retryThreshold:
                            with open(errorFile, "a") as r:
                                print(f"\nWARNING: '{mapped}' not found in concept map, retrying {qId}\n")
                                r.write(f"\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} WARNING: '{mapped}' not found in concept map, retrying {qId}\n")
                                continue
                        else:
                            with open(errorFile, "a") as r:
                                print(f"\nWARNING: '{mapped}' not found in concept map, skipping {qId} after max retries\n")
                                r.write(f"\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} WARNING: '{mapped}' not found in concept map, skipping {qId} after max retries\n")

                print("mapped back to cm:", conceptMapConcepts)

                if len(conceptMapConcepts) < expectedConceptCount:
                    retries += 1
                    if retries < retryThreshold:
                        print(f"\nWARNING: concept mapper returned {len(conceptMapConcepts)}, expected {expectedConceptCount}. Retrying {qId}.\n")
                        with open(errorFile, "a") as r:
                            r.write(f"\n {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} WARNING: concept mapper returned {len(conceptMapConcepts)}, expected {expectedConceptCount}. Retrying {qId}.\n")
                        continue #TODO ideally try get whatever concepts you can
                    else:
                        print(f"\nWARNING: concept mapper returned {len(conceptMapConcepts)}, expected {expectedConceptCount}. Skipping {qId} after max retries.\n")
                        with open(errorFile, "a") as r:
                            r.write(f"\n {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}WARNING: concept mapper returned {len(conceptMapConcepts)}, expected {expectedConceptCount}. Skipping {qId} after max retries.\n")
                        break #TODO ideally try get whatever concepts you can

                classifications = []
                reasonIndex = classificationResponse.find("REASON:")
                index = 0
                while(reasonIndex != -1):
                    conceptIndex = classificationResponse.find("CONCEPT:")
                    levelIndex = classificationResponse.find("LEVEL:")
                    reasoning = classificationResponse[reasonIndex: classificationResponse.find("\n", reasonIndex)].strip()
                    contextItemIndexes = [int(m) for m in re.findall(r'\$\$(\d+)\$\$', reasoning)]
                    #should remove extraneous citations from reasoning thing
                    fileNameList = [mD[m]["fileName"] for m in contextItemIndexes if m < len(mD)]
                    pageNumbers = [mD[m]["page"] for m in contextItemIndexes if m < len(mD)]
                    classifications.append(
                        {
                            "llmIdentifiedConcept": classificationResponse[conceptIndex+9:classificationResponse.find("\n", conceptIndex)].strip(),
                            "conceptMapConcept": conceptMapConcepts[index], #assuming it returns in order
                            "conceptMapId": conceptNamesToConceptIds[conceptMapConcepts[index]],
                            "level": classificationResponse[levelIndex+7: classificationResponse.find("\n", levelIndex)].strip(),
                            "reason": reasoning,
                            "fileNames": fileNameList,
                            "pageNumbers": pageNumbers
                        }
                    )
                    classificationResponse = classificationResponse[reasonIndex+8:]
                    reasonIndex = classificationResponse.find("REASON:")
                    index += 1

                #send question to flask backend
                with open(resultFile, "a") as j:
                    j.write(json.dumps({"classifications":classifications, "questionId": qId, "question": question, "answer": answer})+"\n")

                with open(logFile,"a") as r:
                    r.write(f"\n {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} CMMAPCONCEPTS: {conceptMapConcepts}\n")
                    r.write(f"\nENDCONTEXT---------------------------------------------------------------")

                with open(logFile, "a") as r:
                    r.write(f"\n {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} CONTEXT ------------------------------------------------\n{contextPrompt}")
                    r.write("\n---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

                    success = True
            except Exception as e:
                retries += 1
                if retries < retryThreshold:
                    print(f"Error processing question {qId}: {e}\n Retrying.")
                    with open(logFile, "a") as r:
                        r.write(f" {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Attempt {retries-1} failed. Error processing question {qId}: {e}\n Retrying.")
                        continue
                else:
                    print(f"Error processing question {qId}: {e}\n Skipping {qId}.")
                    with open(logFile, "a") as r:
                        r.write(f" {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Attempt {retries-1} failed. Error processing question {qId}: {e}\n skipping after max retries.")
                        break

                
clientSocket.close()