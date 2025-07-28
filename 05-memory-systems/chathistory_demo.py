"""
Chat History Memory System Demo
===============================

This script demonstrates how to implement conversational memory in AI applications using LangChain.
It creates an Agile Coach chatbot that remembers previous conversations within a session,
enabling natural, contextual interactions where the AI can reference earlier exchanges.

Key Features:
- In-Memory Chat History: Stores conversation messages during the session
- Context Awareness: AI remembers previous questions and answers
- Session Management: Maintains conversation state with session IDs
- Natural Flow: Supports follow-up questions and contextual references

Memory System Components:
1. ChatMessageHistory: Stores messages in memory
2. MessagesPlaceholder: Inserts chat history into prompts
3. RunnableWithMessageHistory: Wraps the chain with memory capabilities

Technical Implementation:
- Uses OpenAI GPT-4o for intelligent responses
- Implements LangChain's memory management system
- Maintains conversation context throughout the session
- Supports unlimited conversation length (memory permitting)

Use Case: Agile Coaching Assistant
- Specialized domain: Agile methodologies and processes
- Contextual guidance: References previous questions and solutions
- Continuous learning: Builds on previous conversation topics

Author: Learn LangChain Course
Date: July 2025
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =============================================================================
# CONFIGURATION & MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Essential for accessing OpenAI's GPT models for conversational AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model for conversation
# Model: GPT-4o - OpenAI's advanced model optimized for:
# - Multi-turn conversations with memory
# - Context understanding and retention
# - Domain expertise (Agile coaching in this case)
# - Natural language generation with consistency
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
# =============================================================================
# MEMORY-AWARE PROMPT ENGINEERING
# =============================================================================

# Create prompt template with memory integration
# This template defines the conversation structure and includes memory placeholders
prompt_template = ChatPromptTemplate.from_messages([
    # System message: Defines the AI's role and expertise
    # Specialized as an Agile Coach for domain-specific knowledge
    ("system", "You are an Agile Coach. Answer any questions "
               "related to the agile process"),
    
    # CRITICAL COMPONENT: MessagesPlaceholder for conversation history
    # Technical Details:
    # - variable_name="chat_history": Maps to the history key in the chain
    # - Automatically inserts all previous messages in chronological order
    # - Provides full conversation context to the model
    # - Enables the AI to reference previous exchanges naturally
    MessagesPlaceholder(variable_name="chat_history"),
    
    # Human message: Current user input
    # The {input} placeholder will be replaced with the user's question
    ("human", "{input}")
])

# Create the basic conversational chain
# Technical Flow: Prompt Template → Language Model → Response
# The | operator creates a pipeline where prompt output feeds into the LLM
chain = prompt_template | llm

# =============================================================================
# MEMORY SYSTEM SETUP
# =============================================================================

# Initialize in-memory chat message history
# ChatMessageHistory characteristics:
# - Storage: RAM-based (temporary, lost on application restart)
# - Performance: Fast read/write operations
# - Scope: Single session, isolated from other conversations
# - Capacity: Limited by available memory
# - Use case: Development, testing, short-lived conversations
history_for_chain = ChatMessageHistory()

# Create memory-enhanced conversational chain
# RunnableWithMessageHistory wraps the basic chain with memory capabilities
# Technical Components:
# 1. chain: The core conversational pipeline (prompt + LLM)
# 2. lambda session_id: history_for_chain: Factory function for history retrieval
#    - Takes session_id as parameter (for multi-user support)
#    - Returns the appropriate ChatMessageHistory instance
#    - In this simple example, always returns the same history object
# 3. input_messages_key: Maps user input to the chain's expected input format
# 4. history_messages_key: Maps chat history to the prompt template variable
chain_with_history = RunnableWithMessageHistory(
    chain,                                    # The base conversational chain
    lambda session_id: history_for_chain,    # History provider function
    input_messages_key="input",               # Key for user input in chain invocation
    history_messages_key="chat_history"       # Key for history in prompt template
)

# =============================================================================
# INTERACTIVE CONVERSATION LOOP
# =============================================================================

# Initialize the Agile coaching session
print("🏃‍♂️ Agile Coach Assistant")
print("Ask me anything about Agile methodologies, processes, and best practices!")
print("I'll remember our conversation context for better assistance.")
print("Type 'quit' or 'exit' to end the session.\n")

# Start continuous conversation loop
# This creates a persistent session where each interaction builds on previous context
while True:
    # Get user input
    # The assistant maintains context across all questions in this session
    question = input("Your Question: ").strip()
    
    # Handle session termination
    if question.lower() in ['quit', 'exit', 'q']:
        print("Thanks for the Agile coaching session! Keep improving! 🚀")
        break
    
    # Process non-empty questions
    if question:
        # Technical Execution Flow:
        # 1. User question is captured
        # 2. Previous chat history is retrieved from memory
        # 3. History + current question are formatted into the prompt template
        # 4. Complete prompt (system + history + current question) sent to LLM
        # 5. LLM generates response with full conversation context
        # 6. Both user question and AI response are automatically saved to memory
        # 7. Response is displayed to user
        
        # Invoke the memory-enhanced chain
        response = chain_with_history.invoke(
            # Input data: Current user question
            {"input": question},
            # Configuration: Session management
            {"configurable": {
                # session_id: Identifies this conversation session
                # In production, you'd use unique IDs for different users
                # Format: user_id, timestamp, or UUID for session isolation
                "session_id": "agile_coaching_session_001"
            }}
        )
        
        # Display the AI response
        # response.content contains the generated text from the language model
        print(f"🤖 Agile Coach: {response.content}\n")
        
        # Optional: Display conversation statistics
        # Uncomment to see memory usage and conversation length
        # print(f"💭 Conversation length: {len(history_for_chain.messages)} messages")