from ollama import Client
from chromadbcollection import PersistentChromaDBCollection
from datasetbuilder import append_training_example
import os
import string
from socket import *
from typing import Any
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
        # print("received rankings: ", rankings)
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

correctDict = {
    "Remember":0,
    "Understand":0,
    "Apply":0,
    "Analyze":0,
    "Evaluate":0,
    "Create":0
}

wrongDict = {
    "Remember":0,
    "Understand":0,
    "Apply":0,
    "Analyze":0,
    "Evaluate":0,
    "Create":0
}

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

with open("qsfocus.txt", "r+") as f:
    for question, category in zip(f,f):
        materialNames = set()
        totalCount += 1
        alreadySearchedIds = set()
        goodContext = False

        topicsListEnd = question.find("}")
        topicsList = question[1:topicsListEnd].split() # space separated, with underscores for spaces in names (for parsing). concept map can have spaces for names

        for i,topic in enumerate(topicsList):
            topicsList[i] = topic.replace("_", " ")

        for topic in topicsList:
            for materialId in conceptNamesToMaterialIdLists[topic]:
                materialNames.add(materialIdsToNames[materialId].strip()) #add fileNames to set, avoiding duplicate file names
    
        materialNamesList = list(materialNames)
        mdFilter = {"fileName":{"$in": materialNamesList}}
        similarChunks = cs16collection.queryCollection([question], 20, mdFilter)
        toRemove = []
        question = question[topicsListEnd+1:]

        focusedPrompt = [{"role":"system", "content":"""You are an expert at analyzing the concepts required to solve an intro CS1 C++ question at a CS1 student level.
                          You will be given a QUESTION that a CS1 student will see on a test. You will rephrase the question to state only the named concepts a student must use to solve the problem.
                          Output this rephrased question on the last line of your repsonse."""},
                          {"role":"user","content": f"QUESTION:{question}"}]
        
        response = qwen.chat(model = "miniqwenbloom2q8", messages = focusedPrompt, think = False)
        print(response["message"]["content"])
        print("-----------------")
        focusedQuery = response["message"]["content"][response["message"]["content"].rfind("\n"):].strip()
        # input()
        similarChunks = contactReRanker(similarChunks, focusedQuery , 2, alreadySearchedIds, mdFilter)

        #leave only top 15
        try:
            similarChunks["ids"][0] = similarChunks["ids"][0][:15]
            similarChunks["metadatas"][0] = similarChunks["metadatas"][0][:15]
            similarChunks["documents"][0] = similarChunks["documents"][0][:15]

        except Exception as e:
            print("Did not find enough relevant queries... skipping this question, error: ", e)
            continue
        # instructorContextPrompt = formPrompt(similarChunks["documents"][0], similarChunks["metadatas"][0], question, True, None)


        #llm reranking is good, but takes way too long
        # with open("beforesortinstructorcontextprompt.txt", mode = "w+") as f:
        #     f.write(instructorContextPrompt)

        # contextMessages = [
        #         {"role":"system", "content": """You are a CS1 instructor evaluating a subset of course material context that would be helpful to a CS1 student solving a given introductory C++ QUESTION.
        #         You will first be given a QUESTION which lists relevant concepts at the start in curly brackets followed by indexed CONTEXT that should be related to the QUESTION. 
        #         Mark unrelated, unhelpful, unclear, or garbled CONTEXT items by putting their index in a list [] in the last line of your response.
        #         If all CONTEXT items are sufficient, the last line of your response will be an empty list [].
        #         Respond quickly. do not think."""},
        #         {"role":"user", "content": instructorContextPrompt}]
        # print("sending to ollama...")
        # response = qwen.chat(model = "miniqwenbloom2q8", messages = contextMessages, think = False)
        # contextResponse = response["message"]["content"]
        # toRemove = list(map(int,contextResponse[contextResponse.rfind("[")+1:contextResponse.rfind("]")].split(",")))
        # for i in toRemove:
        #     similarChunks["ids"][0][i] = -1
        #     similarChunks["metadatas"][0][i] = -1
        #     similarChunks["documents"][0][i] = -1 

        # similarChunks["ids"][0][:] = [x for x in similarChunks["ids"][0] if x != -1]
        # similarChunks["metadatas"][0][:] = [x for x in similarChunks["metadatas"][0] if x != -1]
        # similarChunks["documents"][0][:] = [x for x in similarChunks["documents"][0] if x != -1]

        # with open("aftersortinstructorcontextprompt.txt", mode = "w+") as f:
        #     f.write(formPrompt(similarChunks["documents"][0],similarChunks["metadatas"][0], question, True, None))

        # print(contextResponse)

        # input()

        solutionMessages = [
            {"role":"system", "content":"""You are an average CS1 student that writes short and concise answers to computer science questions.
             You will receive a prompt structured and labeled in this order:
             1. CONTEXT: A small subset of CS1 course materials related to the QUESTION.
             2. QUESTION: A test question from an introductory CS1 C++ course.
            You must list the relevance of all CONTEXT items to the question before writing your answer. If some are not useful, state this.
            Then, answer the QUESTION using the relevant CONTEXT items. Write pseudocode only if the question requires a code solution.
            Then, list the CONTEXT items you used, and if they were direct applications, or supplemented with general knowledge.
            """}
        ]

        mD = similarChunks["metadatas"][0]
        sC = similarChunks["documents"][0]

        contextPrompt = "CONTEXT:"
        for i in range(len(sC)):
            dT = "Handout Page"
            if "Handout" not in mD[i]["fileName"]: dT = "Lecture Page"
            if mD[i]["fileType"] != "pdf": dT = "Code"
            if "HW" in  mD[i]["fileName"]: dT = "Homework"
            contextPrompt += f"\nContext Item {i}:\nFile Name:{mD[i]["fileName"]}\nDocument Type:{dT} \nPage Number:{mD[i]["page"]}\nWeek:{mD[i]["week"]}\n"
            contextPrompt += f"Chunk Text: {sC[i]}"

        questionPrompt = contextPrompt + f"QUESTION: {question}\n"

        #TODO replace above prompt formation with the function, for now i need context prompt separated for debugging in the log files

        firstQuestionPrompt = questionPrompt
        #  f"INSTRUCTIONS: Write a short and concise answer using only concepts within the scope of the CONTEXT."

        print("\n------QUESTION prompt for answer:", question)
        solutionMessages.append({"role":"user","content": firstQuestionPrompt})

        response = qwen.chat(
            model = "miniqwenbloom2q8", messages = solutionMessages, think = False #since we have old repo answers, could skip this part...
        )

        solutionResponse = response["message"]["content"]

        classifyMessages = [{"role":"user","content": firstQuestionPrompt}]

        classifyMessages.append({"role":"assistant", "content":f"ANSWER: {solutionResponse}\n"})

        print(solutionResponse)

        systemPrompt = """You are a helpful assistant that classifies CS1 C++ questions into their most used cognitive level of the Revised Bloom's Taxonomy.

        You will be given an input in this structure:
        1. CONTEXT: A small subset of CS1 course materials related to the QUESTION, enumerated as Context Items.
        2. QUESTION: A test question from an introductory CS1 C++ course.
        3. ANSWER: A possible answer to the QUESTION with analysis of relevant CONTEXT items.

        Use the CONTEXT, ANSWER and the following guidelines to classify.

        COGNITIVE LEVEL DEFINITIONS:
        - Remember: simple recall of syntax, facts or commands.
        - Understand: predicting output of a code segment, simple high-level conceptual understanding, or using language features to evaluate expressions.
        - Apply: direct application of known procedures, familiar algorithms, or similar design principles and paradigms.
        - Analyze: detailed breakdowns of code segments and their purposes, debugging, and correctness of approaches and code.
        - Evaluate: judging code or design against criteria, or comparing two approaches' pros and cons.
        - Create: designing an entirely new algorithm or program previously unseen by students.

        OUTPUT FORMAT:
        First line: "CLASSIFICATION:" , followed by the single highest cognitive level label only.
        Then: three sentences of reasoning citing source file names from the CONTEXT or parts of the ANSWER.
        Do not classify based on keywords associated with the cognitive levels.

        When choosing between Apply and Create, refer to the following rules:
        If the ANSWER combines CONTEXT concepts or algorithms in a way unseen and unfamiliar in the CONTEXT, it is Create.
        If the ANSWER contains multiple references or modifications of logic and concepts in the CONTEXT, or ANY direct applications, it is Apply.
        Create requires higher cognitive load than Apply. Always consider Apply before Create.

        """

        append_training_example("/mldata/stanleyguo/alvin/finetuneqwen/applycreate.jsonl", systemPrompt,contextPrompt,question, solutionResponse)

        classifyMessages.extend([{"role": "system", "content": systemPrompt},
        # examples?
        # {"role":"user", "content": ""}
        ])

        if False:

            secondQuestionPrompt = questionPrompt + f"""INSTRUCTIONS: Classify the QUESTION by analyzing the cognitive level used in the ANSWER in relation to the given CONTEXT using the guidelines above.
            State three sentences of reasoning for your classification and cite sources' file names from the CONTEXT. Mention parts of the ANSWER that influenced your classification.\n"""
            classifyMessages.append({"role":"user", "content": secondQuestionPrompt})
            response = qwen.chat(
                model = "miniqwenbloom2q8", messages = classifyMessages, think = False
            )

            # print(response["message"]["content"])

            # classifyMessages.append({"role":"assistant", "content": response["message"]["content"]})
            # classifyMessages.append({"role":"user", "content":"""Above, a CLASSIFICATION of a QUESTION was given using CONTEXT from course material and an attempted ANSWER that cited use of the CONTEXT. 
            #                          You will verify that the classification does not contradict any CONTEXT cited in the ANSWER.
            #                          You will do this by reading the CONTEXT presented and verifying the CLASSIFICATION does not miss any used CONTEXT items.
            #                          When verifying Apply or Create, if there are any direct applications of CONTEXT, it is Apply.
            #                          State your final, verified classification on the last line of your response."""})

            #if classification is create?

            # classifyMessages.append({"role":"user","content":""""Your job is to find reasons to DOWNGRADE the classification, not confirm it. 
            #                         If the classification is Create, your job is to search the CONTEXT aggressively 
            #                         for any material that could support an Apply classification instead. 
            #                         Only confirm Create if you find absolutely no relevant algorithm or procedure 
            #                         in the CONTEXT. Say only your final classification label in the last line of your response."""})
            # response = qwen.chat(model = "miniqwenbloom2q8", messages = classifyMessages, think = False)
            
            print(response["message"]["content"])
            # input()
            classification = response["message"]["content"][:response["message"]["content"].find("\n")].strip()
            print("--------------------------------------------------FINAL CLASSIFICATION", classification)
            if category.strip() in classification: 
                correctCount += 1
                correctDict[category.strip()] += 1
                logFile = "rightfilererank3.txt"
            else:
                logFile = "wrongfilererank3.txt"
                wrongDict[category.strip()] += 1
        
            with open(logFile, "a+") as f:
                f.write(f"----NUMBER {totalCount}:------\n\n")
                f.write(f"PREDICTED: {classification} / ACUTAL: {category.strip()}\n")
                f.write(f"\nQUESTION:\n {question}\n\n")
                f.write(f"CONTEXT GIVEN:\n\n {contextPrompt}\n\n")
                f.write(f"ANSWER: {response["message"]["content"]}\n\n")
        
    


print(f"----total correct:{correctCount}/{totalCount}")

print(correctDict)
print(wrongDict)

"""
using miniqwenbloom2 q4
bruh 25/41
{'Remember': 0, 'Understand': 4, 'Apply': 11, 'Analyze': 4, 'Evaluate': 1, 'Create': 5}
{'Remember': 1, 'Understand': 4, 'Apply': 3, 'Analyze': 3, 'Evaluate': 1, 'Create': 4}


using miniqwenbloom2 q8

----total correct:31/41
{'Remember': 1, 'Understand': 6, 'Apply': 8, 'Analyze': 6, 'Evaluate': 2, 'Create': 8}
{'Remember': 0, 'Understand': 2, 'Apply': 6, 'Analyze': 1, 'Evaluate': 0, 'Create': 1}

"""


"""
if it needs more context, search thru the embeddings excluding those that were returned the first time and fall below
     a reranker score
retrieval logic:
 only get up to the week the question should appear in (take as much reasoning off the llm as possible so it can focus on classification)
"""