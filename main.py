from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2:1b")

template = """
you are an expert in answering questions about a pizza restaurant

here are some relevant reviews : {reviews}

here is the question to answer: {question}
"""

promt = ChatPromptTemplate.from_template(template)

chain = promt | model

while True:
    print("\n\n----------------------------------")
    question = input("Ask your question (q to quit):")
    print("\n\n")
    
    if question == "q":
        break
    
    reviews = retriever.invoke(question)

    result = chain.invoke({"reviews": [], "question": question})
    print(result)