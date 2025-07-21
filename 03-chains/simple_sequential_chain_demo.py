"""
    Speech Generator using LangChain Sequential Chains

    This application demonstrates a Sequential Chain pattern where:
    1. First chain generates a speech title from a topic
    2. Second chain uses that title to generate a full speech
    3. Both chains are connected sequentially using the pipe operator

    Chain Types Demonstrated:
    - Sequential Chain: Output of first chain becomes input of second chain
    - Simple Chain: Each individual chain is a simple prompt → model → parser flow

    Functional Flow:
    Topic → Title Generation → Speech Generation → Display Results
"""

import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Environment setup - Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the Language Model (LLM)
# Using GPT-4o model for high-quality text generation
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# FIRST CHAIN COMPONENT: Title Generation Prompt
# This prompt template takes a topic and generates an impactful speech title
title_prompt = PromptTemplate(
    input_variables=["topic"],  # Expected input variable
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

# SECOND CHAIN COMPONENT: Speech Generation Prompt  
# This prompt template takes a title and generates a full speech
speech_prompt = PromptTemplate(
    input_variables=["title"],  # Expected input variable from first chain
    template="""You need to write a powerful speech of 350 words
     for the following title: {title}
    """
)

"""
    CHAIN CONSTRUCTION using Pipe Operator (|)
    First Chain: Topic → Title Generation → String Parsing → Display Title → Return Title
    Technical Note: The lambda function serves dual purpose:
    1. st.write(title) displays the title in Streamlit UI
    2. (st.write(title), title)[1] returns the title for next chain (tuple indexing)
"""
first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])

# Second Chain: Title → Speech Generation
# Note: No output parser needed here as we'll access .content from the response
second_chain = speech_prompt | llm

# SEQUENTIAL CHAIN: Combine both chains
# Output of first_chain (title) becomes input for second_chain
final_chain = first_chain | second_chain

# STREAMLIT UI COMPONENTS
st.title("Speech Generator")

# User input field for topic
topic = st.text_input("Enter the topic:")

# EXECUTION LOGIC
if topic:
    # Execute the sequential chain with user input
    # Flow: topic → first_chain (generates title) → second_chain (generates speech)
    response = final_chain.invoke({"topic": topic})
    
    # Display the generated speech
    # Technical Note: response.content contains the actual speech text from the LLM
    st.write("**Generated Speech:**")
    st.write(response.content)