import os
from langchain_openai import ChatOpenAI

"""
    Set OPEN API KEY in your environment variables
    export OPENAI_API_KEY="your_open
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)