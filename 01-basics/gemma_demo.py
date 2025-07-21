from langchain_community.chat_models import ChatOllama

"""
    Gemma is an open-source language model by Google, available via the Ollama platform.
    Install Ollama from https://ollama.com/
    Start the Ollama server: ollama serve
    Download the Gemma model: ollama pull gemma:2b
    Run the model: ollama run gemma:2b
    Then execute this script.
"""
llm=ChatOllama(model="gemma:2b")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)