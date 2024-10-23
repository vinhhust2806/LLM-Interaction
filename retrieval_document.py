from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

embeddings = OllamaEmbeddings(
    model="tinyllama",
)

document = PyPDFLoader('gemma-model.pdf')

pages = document.load_and_split()
print(pages)

vectorstore = DocArrayInMemorySearch.from_documents(   # FAISS.from_documents(pages, embedding=embeddings)
    pages,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is Gemma?")

# show the retrieved document's content
print(retrieved_documents)
