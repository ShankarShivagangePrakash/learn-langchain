"""
Basic Chat Prompt Template Demo (Stateless)
===========================================

This script demonstrates a basic chatbot implementation using LangChain's ChatPromptTemplate
without memory capabilities. Each interaction is independent - the AI has no knowledge of
previous conversations or context from earlier exchanges.

Key Characteristics:
- Stateless Conversations: Each question is processed independently
- No Memory: Previous interactions are not remembered
- Simple Architecture: Direct prompt → LLM → response pipeline
- Streamlit Interface: Web-based user interaction
- Domain Expertise: Specialized Agile coaching assistant

Comparison with Memory Systems:
- ❌ Cannot reference previous questions or answers
- ❌ No conversation continuity or context awareness
- ❌ Users must repeat context in each interaction
- ✅ Simple implementation and lower resource usage
- ✅ No memory management complexity
- ✅ Consistent, predictable responses

Technical Architecture:
1. Static Prompt Template: Fixed system and human message structure
2. Language Model: OpenAI GPT-4o for response generation
3. Streamlit UI: Interactive web interface
4. Direct Chain: Simple prompt → LLM pipeline

Use Cases:
- Simple Q&A systems where context isn't needed
- One-off questions and answers
- Stateless API endpoints
- Educational demonstrations of basic chatbots

Author: Learn LangChain Course
Date: July 2025
"""

import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import ChatPromptTemplate

# =============================================================================
# CONFIGURATION & MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Required for accessing OpenAI's language models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model for stateless conversations
# Model: GPT-4o - OpenAI's advanced model
# Note: Without memory integration, each call is independent
# The model won't remember previous interactions in this session
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
# =============================================================================
# STATELESS PROMPT TEMPLATE SETUP
# =============================================================================

# Create a basic prompt template without memory capabilities
# This template defines the conversation structure but lacks memory integration
prompt_template = ChatPromptTemplate.from_messages([
    # System message: Defines the AI's role and domain expertise
    # Sets the context for Agile coaching but won't remember previous advice given
    ("system", "You are an Agile Coach. Answer any questions "
               "related to the agile process"),
    
    # Human message: Current user input only
    # Key Limitation: Only contains the current question, no conversation history
    # The {input} placeholder will be replaced with the user's current question
    # Missing: MessagesPlaceholder for chat history (unlike memory-enabled version)
    ("human", "{input}")
])

# Note: Comparison with Memory-Enabled Version
# Memory Version: ("system", ...) + MessagesPlaceholder + ("human", "{input}")
# This Version: ("system", ...) + ("human", "{input}") only

# =============================================================================
# STREAMLIT WEB INTERFACE SETUP
# =============================================================================

# Create the main application title
# Streamlit automatically renders this as a large header in the web interface
st.title("🏃‍♂️ Agile Coach Assistant (Stateless)")

# Add informational text about the application's limitations
st.info("💡 **Note**: This is a stateless chatbot. Each question is independent - "
        "I won't remember previous conversations or context.")

# Create input field for user questions
# text_input creates a single-line input widget
# Each interaction is processed independently without context from previous inputs
user_input = st.text_input("Enter your Agile question:", 
                          placeholder="e.g., What is a daily standup meeting?")

# =============================================================================
# STATELESS CONVERSATION PROCESSING
# =============================================================================

# Create the basic conversational chain
# Technical Flow: Prompt Template → Language Model → Response
# No memory components involved - simple linear processing
chain = prompt_template | llm

# Process user input when provided
if user_input:
    # Technical Execution Flow (Stateless):
    # 1. User enters question in Streamlit interface
    # 2. Question is inserted into prompt template {input} placeholder
    # 3. Complete prompt (system message + current question) sent to LLM
    # 4. LLM processes prompt without any conversation history
    # 5. Response generated based solely on current question and system context
    # 6. Response displayed to user
    # 7. No memory storage - interaction is complete and forgotten
    
    # Invoke the stateless chain
    # Key Difference: No session_id or history configuration needed
    # The chain processes only the current input without any context
    response = chain.invoke({"input": user_input})
    
    # Display the AI response
    st.write("🤖 **Agile Coach:**")
    st.write(response.content)
    
    # Demonstrate the statelessness with a warning
    st.warning("⚠️ **Stateless Behavior**: If you ask a follow-up question like "
               "'Can you explain more about that?', I won't know what 'that' refers to. "
               "You'll need to repeat the context in your next question.")
    
    # Optional: Show what the LLM actually received
    with st.expander("🔍 View Prompt Sent to AI"):
        st.code(f"""System: You are an Agile Coach. Answer any questions related to the agile process
        
Human: {user_input}""", language="text")
        
    # Educational comparison with memory systems
    with st.expander("📚 Memory vs Stateless Comparison"):
        st.markdown("""
        **With Memory (like chathistory_demo.py):**
        - ✅ Remembers previous questions and answers
        - ✅ Can handle follow-up questions naturally
        - ✅ Maintains conversation context
        - ✅ References like "it", "that" work correctly
        
        **Without Memory (this demo):**
        - ❌ Each question is independent
        - ❌ No context from previous interactions
        - ❌ Follow-up questions need full context
        - ✅ Simpler implementation
        - ✅ Lower resource usage
        - ✅ No session management needed
        """)

# =============================================================================
# APPLICATION USAGE INSTRUCTIONS
# =============================================================================

st.sidebar.markdown("""
### 🚀 How to Use This App

1. **Ask Questions**: Enter any Agile-related question
2. **Get Answers**: Receive expert coaching advice
3. **No Memory**: Each question is independent

### 📝 Example Questions
- What is Scrum methodology?
- How do I run effective retrospectives?
- What are the roles in an Agile team?
- Explain user story writing best practices

### ⚡ Running This App
```bash
streamlit run chatprompttemplate_demo.py
```

### 🔄 For Memory-Enabled Version
Check out `chathistory_demo.py` to see how conversation memory works!
""")