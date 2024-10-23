from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

embeddings = OllamaEmbeddings(
    model="tinyllama",
)

text = ["Machine learning is a field of AI", "LangChain is the framework for building context-aware reasoning applications" ]

vectorstore = InMemoryVectorStore.from_texts(      # FAISS.from_texts(texts=text, embedding=embeddings)
    text,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
print(retrieved_documents)

# return [Document(page_content='LangChain is the framework for building context-aware reasoning applications'), Document(page_content='Machine learning is a field of AI')]