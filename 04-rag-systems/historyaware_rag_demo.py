"""
History-Aware RAG (Retrieval-Augmented Generation) Demo
======================================================

This script demonstrates an advanced RAG implementation with conversational memory.
It extends the basic RAG concept by maintaining chat history, enabling context-aware
conversations where the system remembers previous exchanges and can reference them
in subsequent interactions.

Key Features:
    - Conversational Memory: Maintains chat history across interactions
    - Context-Aware Retrieval: Uses conversation history to improve document retrieval
    - Streamlit Web Interface: Interactive web-based chat interface
    - Session Management: Handles multiple conversation sessions
    - History-Aware Prompting: Incorporates previous messages in prompt context

Technical Advantages over Basic RAG:
    - Better follow-up question handling
    - Contextual references ("it", "that", "the previous answer")
    - Improved user experience through conversation continuity
    - Session-based memory management

Architecture Components:
    1. Document Processing Pipeline (same as basic RAG)
    2. History-Aware Retriever (enhanced retrieval with chat context)
    3. Message History Management (persistent conversation memory)
    4. Streamlit UI Integration (web-based interface)
    5. Session-based Chain Execution (stateful conversations)

Use Cases:
    - Interactive document exploration
    - Conversational customer support
    - Educational Q&A systems
    - Research assistance with follow-up questions
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings for semantic search
# Same as basic RAG - converts text to vector representations
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize ChatOpenAI for conversational responses
# Model: GPT-4o - optimized for multi-turn conversations
# Enhanced capabilities for maintaining context across interactions
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

"""
=============================================================================
DOCUMENT PROCESSING PIPELINE (SAME AS BASIC RAG)
=============================================================================

    Load product data for conversational Q&A
    Note: This is the same document processing as basic RAG
    The enhancement comes in the retrieval and conversation management
    Fix: Use absolute path to ensure file is found regardless of working directory
"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "product-data.txt")
document = TextLoader(file_path).load()

# Split documents into chunks for vector storage
# Chunk size and overlap optimized for conversational context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Standard chunk size for maintaining context
    chunk_overlap=200   # Overlap ensures conversation continuity
)
chunks = text_splitter.split_documents(document)

# Create vector store for semantic similarity search
# Same vector store as basic RAG - the intelligence is in retrieval logic
vector_store = Chroma.from_documents(chunks, embeddings)

# Initialize basic retriever (enhanced later with history awareness)
retriever = vector_store.as_retriever()

"""
=============================================================================
HISTORY-AWARE PROMPT ENGINEERING
=============================================================================

    Define enhanced prompt template with chat history support
    Key difference from basic RAG: includes MessagesPlaceholder for conversation history
"""
prompt_template = ChatPromptTemplate.from_messages([
    # System message: Same instructions as basic RAG
    ("system", """You are an assistant for answering questions.
    Use the provided context to respond. If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    # CRITICAL ADDITION: MessagesPlaceholder for conversation history
    # This allows the model to see previous exchanges in the conversation
    # Enables contextual understanding of follow-up questions
    MessagesPlaceholder(variable_name="chat_history"),
    
    # Human message: Current user input
    ("human", "{input}")
])

"""
Create history-aware retriever - THE KEY ENHANCEMENT
This retriever considers chat history when searching for relevant documents
Technical Process:
    1. Analyzes current question + chat history
    2. Generates better search queries based on conversation context
    3. Retrieves more contextually relevant document chunks
    4. Enables understanding of references like "it", "that", "the previous answer"
"""
history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

# Create document chain (same as basic RAG but with history-aware prompt)
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# Create RAG chain with history-aware retriever
# This is the enhanced RAG pipeline that maintains conversation context
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

"""
=============================================================================
CONVERSATION MEMORY & SESSION MANAGEMENT
=============================================================================

    Initialize Streamlit-based chat message history
    StreamlitChatMessageHistory provides persistent memory within a Streamlit session
    Features:
    - Automatic message storage and retrieval
    - Integration with Streamlit's session state
    - Maintains conversation context across interactions
"""
history_for_chain = StreamlitChatMessageHistory()

"""
    Create the final chain with conversation memory
    RunnableWithMessageHistory wraps the RAG chain with memory capabilities
    Technical Components:
    - rag_chain: The core RAG functionality with history-aware retrieval
    - lambda session_id: Factory function to get chat history for each session
    - input_messages_key: Maps user input to the chain
    - history_messages_key: Maps chat history to the prompt template
"""
chain_with_history = RunnableWithMessageHistory(
    rag_chain,                           # The enhanced RAG chain
    lambda session_id: history_for_chain,  # Session-based history provider
    input_messages_key="input",          # Key for user input in the chain
    history_messages_key="chat_history"  # Key for chat history in prompt template
)

# =============================================================================
# STREAMLIT WEB INTERFACE FOR CONVERSATIONAL RAG
# =============================================================================

# Create Streamlit web interface header
# Streamlit provides an interactive web-based chat interface
st.write("# 💬 Chat with Document (History-Aware)")
st.write("Ask questions about the product data. The system remembers our conversation!")
st.write("Try follow-up questions like 'Tell me more about it' or 'What about the price?'")

# Create text input widget for user questions
# Streamlit's text_input provides real-time user interaction
question = st.text_input("Your Question:", placeholder="Ask about products, orders, shipping, etc.")

# Process user input through the history-aware RAG pipeline
if question:
    """
        Technical Execution Flow for History-Aware RAG:
        1. User submits question via Streamlit interface
        2. Current question + chat history sent to history-aware retriever
        3. Retriever analyzes conversation context to improve search
        4. Enhanced document retrieval based on conversational context
        5. Retrieved context + chat history + current question sent to LLM
        6. LLM generates contextually aware response
        7. Response displayed in Streamlit + conversation history updated
        
        Invoke the chain with session management
        session_id="abc123": Identifies this conversation session
        In production, you'd use unique session IDs for different users
    """
    response = chain_with_history.invoke(
        {"input": question},                           # Current user question
        {"configurable": {"session_id": "abc123"}}     # Session identifier
    )
    
    # Display the AI response
    st.write("**Assistant:**", response['answer'])

# Instructions for running this Streamlit app
st.sidebar.markdown("""
### How to Run This App
```bash
streamlit run historyaware_rag_demo.py
```

### Example Conversation Flow
1. "What products do you have?"
2. "Tell me more about the smartphone"
3. "What's its price?"
4. "How do I order it?"

### Key Features
- 🧠 **Memory**: Remembers conversation context
- 🔍 **Smart Retrieval**: Uses history for better search
- 💬 **Natural Flow**: Handle follow-up questions
- 🖥️ **Web Interface**: Interactive Streamlit UI
""")

