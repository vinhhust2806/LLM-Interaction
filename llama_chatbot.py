import streamlit as st 
from st_chat_message import message
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = Ollama(model = "tinyllama") # available models are here: https://ollama.com/library

parser = StrOutputParser()

template = """
Answer the question. 
Question: {question}
"""

prompt = PromptTemplate.from_template(template)

chain = prompt | model | parser 

if 'user' not in st.session_state:
     st.session_state.user = []

if 'assistant' not in st.session_state:
     st.session_state.assistant = []

def main():
    question = st.chat_input("Enter your question")
    if question:
          st.session_state.user.append(question)
          st.session_state.assistant.append(chain.invoke({'question': question}).strip())

    for i in range(len(st.session_state.assistant)):   
          message(st.session_state.user[i], is_user=True)
          message(st.session_state.assistant[i])

if __name__ == "__main__":
    main()
