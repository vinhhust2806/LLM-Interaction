from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="tinyllama",  # available models are here: https://ollama.com/library
)

text1 = "Hello"
text2 = "Hi"

two_vectors = embeddings.embed_documents([text1, text2])
