from operator import itemgetter
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

model = Ollama(model = "tinyllama")  # available models are here: https://ollama.com/library
embeddings = OllamaEmbeddings(model = "tinyllama")  # available models are here: https://ollama.com/library

parser = StrOutputParser()

document = PyPDFLoader('gemma-model.pdf')

pages = document.load_and_split()

template = """
Answer the question based on the context below. If you can't 
answer the question, respond with "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

chain = prompt | model | parser 

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding = embeddings)
retriever = vectorstore.as_retriever()

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

questions = [
    "What makes the Gemma model special?",
    "Why is Gemma model a new state-of-the-art?",
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print()

# make the response appear like the style of a chatbot because of a typewriter effect.
for s in chain.stream({"question": "Can I fine-tune Gemma on my own data?"}):
    print(s, end="", flush=True)


# you have a lot of questions to ask and you don't want to wait for the model to process each question one by one. This is done in parallel.
questions = [
    "Can I use TensorFlow and Keras with Gemma?",
    "Is there debugging support?",
]

print(chain.batch([{"question": q} for q in questions]))

