from ollama import Client


qwen = Client(host = "localhost:11434")

messages = [
    {"role": "system", "content": "You are a helpful assistant that classifies computer science questions into the Revised Bloom's Taxonomy.\
      The questions come from an introductory CS1 course that teaches C++ up to basic linked lists and recursion. You will be given\
     context regarding the question you are to classify via a RAG system that queries a vector database of embeddings of course handouts, lecture slides, and homeworks, embedded by the jina-code-embeddings-1.5b model. The context should influence your classification\
     significantly. Regarding the materials themselves, handouts and lecture slides are given during class, and students work through handout problems together and with the help of the instructor. Homeworks are completed asynchronously, and students can receive help from office hours."},
     {"role" :"user", "content" : "In the next message, I am going to give you only the structured_query response from a Neo4jPropertyGraphStore instance of the llamaindex framework.\
             You will analyze the response and outline the graph in these ways: 1. How many nodes are there, and what are their labels? 2. How many edges are there, and what are their labels?\
                 3. List each triplet in the given graph. 4. Why did carlo implement von? Answer with graph context."},
    {"role": "user","content": f"{response}"}

]