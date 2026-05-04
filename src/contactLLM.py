from ollama import Client
from chromadbcollection import PersistentChromaDBCollection
import os

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

question = ""

messages = [
    {"role": "system", "content": """You are a helpful assistant that classifies computer science questions into the highest required cognitive level of the Revised Bloom's Taxonomy.
The questions come from an introductory CS1 C++ course. You will be given context from course materials that the students are familiar with. Use the context and the following guidelines to classify.

COGNITIVE LEVEL DEFINITIONS:
- Remember: simple recall of syntax or facts.
- Understand: predicting output of a code segment, simple high-level conceptual understanding, or using language features to evaluate expressions.
- Apply: application of known procedures or familiar algorithms.
- Analyze: detailed breakdowns of code segments and their purposes.
- Evaluate: judging code or design against criteria, or comparing two approaches' pros and cons.
- Create: designing an entirely new algorithm or program previously unseen by students.

OUTPUT FORMAT:
First line: the single highest cognitive level label only.
Then: three sentences of reasoning citing source file names from the CONTEXT.
If the CONTEXT is unclear, still classify but note your uncertainty.
Do not classify based on keywords associated with the cognitive levels.

CRITICAL RULE - APPLY vs CREATE:
Variable names do NOT define an algorithm. Two functions are the SAME algorithm if they perform identical operations in the same logical order, regardless of variable names.
f(a, b%a) and f(b, a%b) are the SAME algorithm — just swapped variable names.
A slight variation, reordering, or renaming of an algorithm from the CONTEXT is still Apply.
For example, if baseConversion(n,m) appears in the CONTEXT but the QUESTION asks for baseConversion(m,n), it is Apply.
Do NOT classify as Create unless the algorithm is entirely absent from the CONTEXT and cannot be derived by any simple modification of material in the CONTEXT.
When in doubt between Apply and Create (and NO OTHER SITUATIONS), choose Apply if the CONTEXT provided an example or explanation of the algorithm.
"""},


    {"role": "user", "content": "CONTEXT:\nContext Item 1: The GCD of x and y is recursively computed as GCD(x, y%x).\nQUESTION: Write a function GCD(y, x%y) that computes the greatest common divisor."},
    {"role": "assistant", "content": "Apply\nOne might initially think GCD(y, x%y) differs from GCD(x, y%x) because the variable names are swapped, but variable names are irrelevant to algorithm identity. The recursive structure and modulo operation are identical to Context Item 1. Since this algorithm appeared in the context under a simple variable swap, this is Apply not Create."},

    {"role": "user", "content": "CONTEXT:\nContext Item 1: The GCD of x and y is recursively computed as GCD(x, y%x).\nQUESTION: Design a function that finds the LCM of two numbers without using GCD."},
    {"role": "assistant", "content": "Create\nThe question asks for an LCM algorithm that explicitly avoids GCD, which does not appear in the context in any form. No variable substitution or minor modification of Context Item 1 would produce an LCM algorithm. This requires designing a new procedure entirely absent from the course materials."},
]

similarChunks = cs16collection.queryCollection([question], 5)

mD = similarChunks["metadatas"][0]
sC = similarChunks["documents"][0]
documentsAndData = []
questionPrompt = "CONTEXT:"
for i in range(len(sC)):
    dT = "Handout Page"
    if "Handout" not in mD[i]["fileName"]: dT = "Lecture Page"
    if mD[i]["fileType"] != "pdf": dT = "Code"
    if "HW" in  mD[i]["fileName"]: dT = "Homework"
    questionPrompt += f"\nContext Item {i+1}:\nFile Name:{mD[i]["fileName"]}\nDocument Type:{dT} \nPage Number:{mD[i]["page"]}\nWeek:{mD[i]["week"]}\n"
    questionPrompt += f"Chunk Text: {sC[i]}"

questionPrompt += f"QUESTION: {question}\n"
questionPrompt += f"INSTRUCTIONS: Classify the QUESTION using the CONTEXT above. State three sentences of reasoning for your classification and cite sources' file names from the CONTEXT. If the CONTEXT is unclear, make a classification but state that your classification is unsure.\n"
print(questionPrompt)
messages.append({"role":"user","content": questionPrompt})
response = qwen.chat(
    model = "qwenbloom2", messages = messages, think = False
)
print(response["message"]["content"])

"""
if it needs more context, search thru the embeddings excluding those that were returned the first time and fall below
     a reranker score
retrieval logic:
 only get up to the week the question should appear in (take as much reasoning off the llm as possible so it can focus on classification)
"""