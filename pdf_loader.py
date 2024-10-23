from langchain_community.document_loaders import PyPDFLoader

document = PyPDFLoader('gemma-model.pdf')

pages = document.load_and_split()