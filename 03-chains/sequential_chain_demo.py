"""
    Advanced Sequential Chain Demo - Speech Generator with Emotion Control

    FUNCTIONAL OVERVIEW:
    This application demonstrates a more complex Sequential Chain pattern where:
    1. First chain: Generates a speech title from a user-provided topic
    2. Lambda transformer: Combines the generated title with user-provided emotion
    3. Second chain: Uses both title AND emotion to generate a customized speech

    KEY DIFFERENCES from Simple Sequential Chain:
        - Multiple input variables (topic + emotion)
        - Data transformation between chains using lambda functions
        - Dictionary-based data passing between chain components

    CHAIN FLOW:
    Topic Input → Title Generation → Data Transformation → Emotional Speech Generation → Display

    TECHNICAL CONCEPTS DEMONSTRATED:
        - Sequential chaining with data transformation
        - Lambda functions for data restructuring
        - Multi-variable prompt templates
        - Streamlit UI integration with multiple inputs
"""

import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Environment Configuration
# Retrieve OpenAI API key from environment variables for secure access
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Language Model Initialization
# Using GPT-4o for high-quality text generation with advanced reasoning capabilities
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# FIRST CHAIN COMPONENT: Title Generation Prompt Template
# Takes only 'topic' as input and generates a compelling speech title
title_prompt = PromptTemplate(
    input_variables=["topic"],  # Single input variable
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

# SECOND CHAIN COMPONENT: Emotional Speech Generation Prompt Template
# Takes TWO inputs: 'title' (from first chain) and 'emotion' (from user)
speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],  # Multiple input variables
    template="""You need to write a powerful {emotion} speech of 350 words
     for the following title: {title}   
    """
)

"""
    CHAIN CONSTRUCTION WITH DATA TRANSFORMATION

    First Chain: Topic → Title Generation → String Parsing → UI Display → Title Return
    Technical Note: Lambda function serves dual purpose:
    1. st.write(title) - displays generated title in Streamlit UI
    2. (st.write(title), title)[1] - returns title string for next chain (tuple indexing trick)
"""
first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])

# Second Chain: Dictionary Input → Speech Generation
# Expects input format: {"title": "...", "emotion": "..."}
second_chain = speech_prompt | llm

"""
    COMPLEX SEQUENTIAL CHAIN WITH DATA TRANSFORMATION
    Critical Technical Component: Lambda transformer function
    Purpose: Convert single title string to dictionary format required by second chain
    Input: title (string from first_chain)
    Output: {"title": title, "emotion": emotion} (dictionary for second_chain)
    Note: 'emotion' variable must be in scope when this lambda executes
"""
final_chain = first_chain | (lambda title: {"title": title, "emotion": emotion}) | second_chain

# STREAMLIT USER INTERFACE
st.title("Speech Generator")

# User Input Collection
# Two separate inputs required for this advanced chain
topic = st.text_input("Enter the topic:")  # Input for first chain
emotion = st.text_input("Enter the emotion:")  # Input for second chain (combined with title)

# EXECUTION LOGIC WITH VALIDATION
# Both inputs are required before chain execution
if topic and emotion:
    # Execute the complex sequential chain
    # Flow: {"topic": topic} → first_chain → title → lambda transformer → 
    #       {"title": title, "emotion": emotion} → second_chain → emotional speech
    response = final_chain.invoke({"topic": topic})
    
    # Display the final generated speech
    # Technical Note: response.content contains the LLM's text output
    st.write("**Generated Emotional Speech:**")
    st.write(response.content)

