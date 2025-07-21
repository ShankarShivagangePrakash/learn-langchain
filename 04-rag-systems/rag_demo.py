"""
RAG (Retrieval-Augmented Generation) Demo
=========================================

This script demonstrates a basic RAG implementation using LangChain.
RAG combines document retrieval with language generation to answer questions
based on specific documents rather than just the LLM's training data.

Architecture:
    1. Load and chunk documents
    2. Create embeddings and store in vector database
    3. Set up retrieval system
    4. Combine retrieval with language model for Q&A
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings model
# Purpose: Convert text chunks into high-dimensional vectors for semantic similarity search
# Model: text-embedding-ada-002 (OpenAI's latest embedding model)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize the language model for generation
# Model: GPT-4o - OpenAI's advanced model for natural language understanding and generation
# Purpose: Generate human-like responses based on retrieved context
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

"""
=============================================================================
DOCUMENT PROCESSING PIPELINE
=============================================================================

    Step 1: Load the document
    TextLoader reads the product FAQ data from a text file
    The document contains product information, order details, and customer support FAQs"""
document = TextLoader("product-data.txt").load()

"""
    Step 2: Split document into chunks
    RecursiveCharacterTextSplitter breaks large documents into smaller, manageable pieces
    chunk_size=1000: Each chunk contains approximately 1000 characters
    chunk_overlap=200: 200 characters overlap between chunks to maintain context continuity
    This prevents important information from being split across chunks
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)

"""
    Step 3: Create vector store with embeddings
    Chroma is an open-source vector database for storing and searching embeddings
    from_documents() creates embeddings for all chunks and stores them in the database
    This enables semantic similarity search rather than just keyword matching
"""
vector_store = Chroma.from_documents(chunks, embeddings)

"""
    Step 4: Create retriever
    The retriever handles searching the vector store for relevant chunks
    It finds documents most similar to the user's query using cosine similarity
"""
retriever = vector_store.as_retriever()

"""
=============================================================================
PROMPT ENGINEERING & CHAIN SETUP
=============================================================================

    Define the prompt template for the RAG system
    This template structures how the LLM receives context and user queries
"""
prompt_template = ChatPromptTemplate.from_messages([
    # System message: Defines the AI assistant's behavior and constraints
    ("system", """You are an assistant for answering questions.
    Use the provided context to respond. If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    # Human message: Contains the user's actual question
    ("human", "{input}")
])


"""
    Create the document chain
    This chain takes the retrieved documents and combines them with the LLM
    "stuff" strategy means all retrieved docs are stuffed into the prompt context
"""
qa_chain = create_stuff_documents_chain(llm, prompt_template)

"""
    Create the complete RAG chain
    This combines the retriever (finding relevant docs) with the QA chain (generating answers)
    Flow: User Query → Retriever finds relevant docs → QA chain generates answer using docs
"""
rag_chain = create_retrieval_chain(retriever, qa_chain)

"""
=============================================================================
INTERACTIVE Q&A INTERFACE
=============================================================================

    Start the interactive chat session
"""
print("Chat with Document")
print("Ask questions about the product data. Type your question below:")

# Get user input
# The user can ask questions about products, orders, shipping, etc.
question = input("Your Question: ")

# Process the question through the RAG pipeline
if question:
    """Technical Flow:
        1. User question is converted to embeddings
        2. Retriever searches vector store for similar chunks
        3. Retrieved chunks are formatted as context
        4. Context + question are sent to LLM via prompt template
        5. LLM generates response based on retrieved context
    """
    response = rag_chain.invoke({"input": question})
    
    """Display the generated answer
        The response dictionary contains:
        - 'answer': The generated response
        - 'context': The retrieved document chunks used
        - 'input': The original question
    """
    print("\nResponse:", response['answer'])