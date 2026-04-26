from ollama import Client
from chromadbcollection import PersistentChromaDBCollection
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

qwen = Client(host = "localhost:11434")

question = " FUNCTIONS CODING RECURSIVE The Towers of Hanoi is a classic problem in computer science and mathematics. The objective of the\
problem is to move a set of disks from one peg to another, following a set of rules:\
- Only one disk can be moved at a time.\
- A disk can only be placed on top of a larger disk or an empty peg.\
- All disks must be moved in the same order from the source peg to the destination peg.Example:You have three pegs (A, B, and C) and a set of n disks of different sizes. Initially, all n disks are\
stacked in descending order of size (smallest on top) on peg A. The goal is to move all the disks\
from peg A to peg C, using peg B as an auxiliary peg.towersOfHanoi(3, 'A', 'C', 'B');Output:Move disk 1 from A to C\
Move disk 2 from A to B\
Move disk 1 from C to B\
Move disk 3 from A to C\
Move disk 1 from B to A\
Move disk 2 from B to C\
Move disk 1 from A to C\
Write ”void towersOfHanoi(int n, char source, char destination, char auxiliary);” function using a recursive approach.Hint: In every step try to find the largest disk left (from the n-1 subset) to stack it on peg C."

question = "Create a recursive function to calculate the Greatest Common Divisor (GCD) of two integers using thefollowing theorem.GCD Theorem: Given two integers x and y, if x divides y, x is the GCD of x and y.If x doesn’t divide y, GCD of x and y is given by GCD of x and y mod x."

question = "Write a function isPalindrome that takes a C string as input and returns true if the string is a palindrome and false otherwise. "
# question = "Create a recursive function to calculate the Greatest Common Divisor (GCD) of two integers using the\
# following theorem.GCD Theorem: Given two integers a and b, if a divides b, a is the GCD of a and b.\
# If a doesn’t divide b, GCD of a and b is given by GCD of a and b mod a."

question = "You have collected unsorted data in a vector. In this problem, you will write 3 functions to analyze the data. How is the parameter data passed into these functions? Why might this choice be useful in the context of analyzing data?"

# question = "What is the return value of calling factorial(4)?"

question  = "Consider the following ‘Enemy’ struct and ‘generate_enemy’ function. Noticed that an ‘Enemy’ struct is created on the stack in the ‘generate_enemy’ function. Explain the potential downsides of this and why using the heap would be better. "

question = "Write a function that prints the quotient and remainder of a numerator given its divisor."
question = "You have collected unsorted data in a vector. In this problem, you will write 3 functions to analyze the data. What is the meaning of the const keyword in the function declarations above?"

# question = "Create a recursive function to calculate the Greatest Common Divisor (GCD) of two integers using the\
# following theorem.GCD Theorem: Given two integers x and y, if y divides x, y is the GCD of x and y.\
# If y doesn’t divide x, GCD of x and y is given by GCD of y and x mod y."

# question = "Given the following 3x3 matrix, your task is to modify the matrix by replacing every occurrence of the number ’2’ with ’12’. Use two lines of code that modify the occurrences of '2' in the matrix"

cs16collection = PersistentChromaDBCollection("localhost:11434", 
                                              "jinacpu", #"huggingface.co/jinaai/jina-code-embeddings-1.5b-GGUF:latest"
                                              "../data/persistent",
                                              "cs16collection")
persistentPath = "../cs16materials"
status = cs16collection.parseAndPopulate(persistentPath)

if status == 0: print(f"Found existing persistent chromadb collection at {persistentPath}")


messages = [
    {"role": "system", "content": "You are a helpful assistant that classifies computer science questions into the highest required cognitive level of the Revised Bloom's Taxonomy.\
      Do not let keywords like 'Create' or 'Evaluate' entirely influence your decision. The questions come from an introductory CS1 C++ course. You will be given\
     context regarding the question you are to classify that comes from course material such as homeworks, lecture slides, and lecture handouts. The context should influence your classification\
     significantly. Regarding the materials themselves, handouts and lecture slides are given during class, and students work through handout problems together and with the help of the instructor.\
     A question that asks for simple recall of syntax or facts is Remember level.\
     A question that asks for output of a code segment or simple, high level conceptual understanding is Understand level.\
     A question that asks for application of known procedures or previously known algorithms is Apply level.\
     To distinguish between Apply and Create, use these guidelines:\
     If the algorithm and its implementation or a slight variation appear in the CONTEXT, it is Apply, because\
     the students are familiar with the algorithm and process so the cognitive load is lower and therefore the question should be categorised as Apply. \
     Different or swapped variable names in the CONTEXT do not imply it is a different algorithm in the QUESTION, and should still be categorised as Apply.\
     A question that asks for detailed breakdowns of code segments and their purposes is Analyze level.\
     A question that asks to judge code or design against criteria, or comparing two approaches' pros and cons is Evaluate level.\
     A question that asks for synthesis of prior known concepts into a new functional whole that has never been seen by students is Create level.\
     Your classification will label the highest involved process in the question in the first line of your response, taking into account the context given."},
]


"""
     While they haven’t seen the algorithm before, they might have seen background material or bits and pieces, but not the completed whole. \
     The cognitive category of Apply requires knowledge of an algorithm and/or process and its application to a given situation. \
     In programming terms this is where students have seen the same or a very similar algorithm working with different data or presented in a different implementation language.
     The cognitive category of Create applies where the student has no familiarity with completed functional whole.
    If prerequisite topics needed in the algorithm appear in the CONTEXT but the QUESTION algorithm itself hasn't been covered, it is Create.\
     The Create category should require creative thinking and the formation of a coherent or functional whole.
     """
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
questionPrompt += f"INSTRUCTIONS: Classify the QUESTION using the CONTEXT above. State two sentences of reasoning for your classification and cite sources' file names from the CONTEXT. If the CONTEXT is unclear, make a classification but state that your classification is unsure.\n"
print(questionPrompt)
messages.append({"role":"user","content": questionPrompt})
response = qwen.chat(
    model = "qwenbloom", messages = messages, think = False
) #qwen3.5:35b-a3b
print(response["message"]["content"])

"""
if it needs more context, search thru the embeddings excluding those that were returned the first time
retrieval logic:
 only get up to the week the question should appear in (take as much reasoning off the llm as possible so it can focus on classification)
"""