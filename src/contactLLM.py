from ollama import Client
from chromadbcollection import PersistentChromaDBCollection
import os
import string
from socket import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

qwen = Client(host = "localhost:11434")

cs16collection = PersistentChromaDBCollection("localhost:11434", 
                                              "jinacpu", #"huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest"
                                              "../data/persistent",
                                              "cs16collection")
persistentPath = "../cs16materials"
status = cs16collection.parseAndPopulate(persistentPath)

if status == 0: print(f"Found existing persistent chromadb collection at {persistentPath}")


def contactReRanker(similarChunks, question): #add support for full TCP comms or dump to a file on a shared system
    serverName = '127.0.0.1'
    serverPort = 2020
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))
    toRemove = string.whitespace.replace(' ', '') #TODO verify method 
    table = str.maketrans('', '', toRemove)
    with open('reranker/contexts.txt', 'a+') as f:
        for i, sC in enumerate(similarChunks["documents"][0]):
            f.write(f"{similarChunks["ids"][0][i]}:{sC.translate(table)}\n")
        f.write(question.translate(table))
    sentence = "contexts.txt$EOM$"
    # while(sentence != '@'): implement persistent connection to avoid tcp overhead and slowstart (but the constant cost is very small at this point)
    clientSocket.send(sentence.encode())
    serverResponse = clientSocket.recv(1024)
    print("From server:", serverResponse.decode())
    serverResponse = clientSocket.recv(1024) #server reranking, or some error
    print("serverResponse", serverResponse)
    clientSocket.close()


with open("qs.txt", "r+") as f:
    for question, category in zip(f,f):
        print("QUESTION----", question)
        similarChunks = cs16collection.queryCollection([question], 8)
        contactReRanker(similarChunks, question)
        # print(similarChunks)
        # mD = similarChunks["metadatas"][0]
        # sC = similarChunks["documents"][0]
        # scores = similarChunks["distances"][0]
        # ids = similarChunks["ids"][0]
        # for index, chunk in enumerate(sC):
        #     print("Score:")
        #     print(scores[index])
        #     print("Metadata")
        #     print(mD[index])
        #     print("--------------------------------------")
        #     print("Context:")
        #     print(chunk)
        
        input()


while True:
    pass
correctCount = 0
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

with open("qs.txt", "r+") as f:
    for question, category in zip(f,f):
        totalCount +=1
        solutionMessages = [
            {"role":"system", "content":"""You are a CS1 student that writes short and concise answers to computer science questions. Write pseudocode only if the question requires a code solution.
            The questions come from an introductory CS1 C++ course. You will be given context from course materials that the students are familiar with. List the contex items used under Used Contexts at the end of your answer.
             Use concepts from the CONTEXT to form your answer and cite specific CONTEXT lines you used verbatim, if any."""}
        ]

        similarChunks = cs16collection.queryCollection([question], 5)

        mD = similarChunks["metadatas"][0]
        sC = similarChunks["documents"][0]
        documentsAndData = []
        contextPrompt = "CONTEXT:"
        for i in range(len(sC)):
            dT = "Handout Page"
            if "Handout" not in mD[i]["fileName"]: dT = "Lecture Page"
            if mD[i]["fileType"] != "pdf": dT = "Code"
            if "HW" in  mD[i]["fileName"]: dT = "Homework"
            contextPrompt += f"\nContext Item {i+1}:\nFile Name:{mD[i]["fileName"]}\nDocument Type:{dT} \nPage Number:{mD[i]["page"]}\nWeek:{mD[i]["week"]}\n"
            contextPrompt += f"Chunk Text: {sC[i]}"

        questionPrompt = contextPrompt + f"QUESTION: {question}\n"

        firstQuestionPrompt = questionPrompt + f"INSTRUCTIONS: Write a short and concise answer using only concepts within the scope of the CONTEXT."

        # print(firstQuestionPrompt)
        print("\n------question prompt for answer:", question)
        solutionMessages.append({"role":"user","content": firstQuestionPrompt})
        response = qwen.chat(
            model = "miniqwenbloom2q8", messages = solutionMessages, think = False #since we have old repo answers, could skip this part...
        )

        solutionResponse = response["message"]["content"]

        classifyMessages = [{"role":"user","content": firstQuestionPrompt}]

        classifyMessages.append({"role":"assistant", "content":f"ANSWER_FOR_QUESTION: {solutionResponse}\n"})

        print(solutionResponse)

        classifyMessages.extend([{"role": "system", "content": """You are a helpful assistant that classifies computer science questions into their most used cognitive level of the Revised Bloom's Taxonomy.
        The questions come from an introductory CS1 C++ course. You will be given CONTEXT from course materials that the students are familiar with, and ANSWER_FOR_QUESTION that describes a CS1 student's answer.
                                Use the CONTEXT, ANSWER_FOR_QUESTION and the following guidelines to classify.

        COGNITIVE LEVEL DEFINITIONS:
        - Remember: simple recall of syntax, facts or commands.
        - Understand: predicting output of a code segment, simple high-level conceptual understanding, or using language features to evaluate expressions.
        - Apply: application of known procedures or familiar algorithms.
        - Analyze: detailed breakdowns of code segments and their purposes, debugging, and correctness of approaches.
        - Evaluate: judging code or design against criteria, or comparing two approaches' pros and cons.
        - Create: designing an entirely new algorithm or program previously unseen by students.

        OUTPUT FORMAT:
        First line: the single highest cognitive level label only.
        Then: three sentences of reasoning citing source file names from the CONTEXT or parts of the ANSWER_FOR_QUESTION.
        Do not classify based on keywords associated with the cognitive levels.
            
        When choosing between Apply and Create, refer to the following rules:
        If the ANSWER_FOR_QUESTION contains multiple references or modifications of logic and concepts in the CONTEXT, it is Apply.
        If the ANSWER_FOR_QUESTION combines CONTEXT concepts or algorithms in a way unseen and unfamiliar in the CONTEXT, it is Create.
        Create requires higher cognitive load than Apply.

        """},
        # examples?
        # {"role":"user", "content": ""}
        ])
        secondQuestionPrompt = questionPrompt + f"""INSTRUCTIONS: Classify the QUESTION by analyzing the cognitive level used in the ANSWER_FOR_QUESTION in relation to the given CONTEXT.
        State three sentences of reasoning for your classification and cite sources' file names from the CONTEXT. Mention parts of the ANSWER_FOR_QUESTION that influenced your classification.\n"""
        classifyMessages.append({"role":"user", "content": secondQuestionPrompt})
        response = qwen.chat(
            model = "miniqwenbloom2q8", messages = classifyMessages, think = False
        )

        print(response["message"]["content"])
        classification = response["message"]["content"][:response["message"]["content"].find("\n")].strip()
        if classification == category.strip(): 
            correctCount += 1
            correctDict[category.strip()] += 1
            logFile = "rightfile.txt"
        else:
            logFile = "wrongfile.txt"
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