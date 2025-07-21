import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.globals import set_debug

# sets the debug mode for LangChain
set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

"""
    Streamlit is an open-source app framework for Machine Learning and Data Science projects.
    It will provide UI for the user to interact with the model.
    To run this app, use the command: streamlit run <python_file>.py
"""
st.title("Ask Anything")

# Provides text input for the user to enter their question
question = st.text_input("Enter the question:")

if question:
    # Invokes the language model with the user's question and displays the response
    response = llm.invoke(question)
    st.write(response.content)