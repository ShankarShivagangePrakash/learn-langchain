import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

"""
    Prompt Template is a way to define a template for the input to the LLM.
    It allows you to define a template with placeholders for the input variables.
    In the below template, there are three input variables: country, no_of_paras, and language.
    The template is used to generate a prompt for the LLM to answer the question about traditional cuisine of a specific country.
"""
prompt_template = PromptTemplate(
    input_variables=["country","no_of_paras","language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}
    """
)

st.title("Cuisine Info")

country = st.text_input("Enter the country:")
no_of_paras = st.number_input("Enter the number of paras",min_value=1,max_value=5)
language = st.text_input("Enter the language:")

if country:
    # Invoke the LLM with the prompt template by passing the input variables
    response = llm.invoke(prompt_template.format(country=country,
                                                 no_of_paras=no_of_paras,
                                                 language=language
                                                 ))
    st.write(response.content)