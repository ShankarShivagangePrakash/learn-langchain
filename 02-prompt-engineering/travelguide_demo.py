import streamlit as st

"""
    A Streamlit web application that generates a personalized travel guide for a user-specified city, month, language and budget.
        - Uses LangChain's PromptTemplate to dynamically construct a travel guide prompt.
        - Employs the ChatOllama Large language model (with the 'llama3.2' model) to generate responses.
        - Collects user input for city, month, language, and budget via Streamlit widgets.
        - Displays the AI-generated travel guide in the Streamlit interface.
    
    Technical details:
        - Requires 'streamlit', 'langchain', and 'langchain_community' packages.
        - The LLM is invoked only when all input fields are provided.
"""
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama

llm=ChatOllama(model="llama3.2")
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)

st.title("Travel Guide")

city = st.text_input("Enter the city:")
month = st.text_input("Enter the month of travel")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget",["Low","Medium","High"])


if city and month and language and budget:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month=month,
                                                 language=language,
                                                 budget=budget
                                                 ))
    st.write(response.content)