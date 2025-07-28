"""
Streamlit Chat History Memory System Demo
=========================================

This script demonstrates how to implement conversational memory in Streamlit web applications
using LangChain's StreamlitChatMessageHistory. It creates a memory-enabled Agile Coach chatbot
that remembers conversations within the Streamlit session and displays the conversation history.

Key Features:
- Streamlit-Integrated Memory: Uses Streamlit's session state for persistence
- Web-Based Interface: Interactive browser-based chat experience
- Session Persistence: Memory survives page refreshes within the same session
- Conversation Visualization: Shows complete chat history to users
- Context Awareness: AI remembers previous exchanges for natural conversation flow

Technical Advantages over Basic In-Memory Systems:
- Session State Integration: Leverages Streamlit's built-in session management
- Web Application Ready: Designed specifically for web deployment
- User-Friendly Interface: Visual conversation history display
- Session Isolation: Different browser sessions have separate memories
- Persistence Across Refreshes: Memory survives page reloads

Memory System Components:
1. StreamlitChatMessageHistory: Streamlit-specific memory storage
2. MessagesPlaceholder: Inserts chat history into prompts
3. RunnableWithMessageHistory: Wraps the chain with memory capabilities
4. Session State Management: Streamlit handles session persistence

Comparison with chathistory_demo.py:
- Similar core functionality but optimized for Streamlit
- Visual conversation history display
- Better user experience with web interface
- Session state persistence vs in-memory storage

Use Case: Web-Based Agile Coaching Assistant
- Accessible via web browser
- Persistent conversations during session
- Visual feedback of conversation flow
- Professional deployment-ready interface

Author: Learn LangChain Course
Date: July 2025
"""

import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =============================================================================
# CONFIGURATION & MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Essential for accessing OpenAI's GPT models for conversational AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model for web-based conversations
# Model: GPT-4o - optimized for multi-turn conversations with excellent context handling
# Perfect for Streamlit applications requiring consistent, contextual responses
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
# =============================================================================
# STREAMLIT-OPTIMIZED PROMPT TEMPLATE WITH MEMORY
# =============================================================================

# Create prompt template with Streamlit session-aware memory integration
# Designed specifically for web applications with persistent session state
prompt_template = ChatPromptTemplate.from_messages([
    # System message: Defines the AI's role and expertise domain
    # Establishes context for Agile coaching throughout the web session
    ("system", "You are an Agile Coach. Answer any questions "
               "related to the agile process"),
    
    # CRITICAL COMPONENT: MessagesPlaceholder for Streamlit session history
    # Technical Details:
    # - Integrates with Streamlit's session state management
    # - Automatically retrieves conversation history from the current browser session
    # - Maintains context across page refreshes within the same session
    # - Enables natural conversation flow in web interface
    MessagesPlaceholder(variable_name="chat_history"),
    
    # Human message: Current user input from Streamlit interface
    # The {input} placeholder receives text from st.text_input widget
    ("human", "{input}")
])

# Create the conversational chain for web deployment
# Technical Pipeline: Streamlit Input → Prompt Template → LLM → Web Display
chain = prompt_template | llm

# =============================================================================
# STREAMLIT-SPECIFIC MEMORY SYSTEM SETUP
# =============================================================================

# Initialize Streamlit-integrated chat message history
# StreamlitChatMessageHistory advantages over basic ChatMessageHistory:
# - Session State Integration: Uses st.session_state for automatic persistence
# - Web Application Optimized: Designed specifically for Streamlit apps
# - Browser Session Isolation: Each browser tab/session has separate memory
# - Refresh Persistence: Survives page refreshes within the same session
# - Multi-User Support: Different users get isolated conversation histories
history_for_chain = StreamlitChatMessageHistory()

# Create memory-enhanced chain with Streamlit session management
# RunnableWithMessageHistory configured for web application deployment
# Technical Components:
# 1. chain: The core conversational pipeline optimized for web interaction
# 2. lambda session_id: history_for_chain: Session factory for web sessions
#    - In web apps, session_id could be based on user authentication
#    - For demo purposes, uses a fixed session ID
# 3. input_messages_key: Maps Streamlit input to chain processing
# 4. history_messages_key: Maps session history to prompt template
chain_with_history = RunnableWithMessageHistory(
    chain,                                    # Web-optimized conversational chain
    lambda session_id: history_for_chain,    # Streamlit session history provider
    input_messages_key="input",               # Key for web form input
    history_messages_key="chat_history"       # Key for session history integration
)

# =============================================================================
# STREAMLIT WEB INTERFACE WITH MEMORY VISUALIZATION
# =============================================================================

# Create main application header with memory indication
st.title("🏃‍♂️ Agile Coach Assistant (Memory-Enabled)")

# Add informational banner about memory capabilities
st.info("🧠 **Memory Enabled**: I remember our conversation throughout this session! "
        "Try asking follow-up questions that reference previous topics.")

# Create input field for user questions
# Streamlit text_input widget captures user interaction
user_input = st.text_input("Ask your Agile question:", 
                          placeholder="e.g., What is Scrum? (I'll remember for follow-ups)")

# Process user input with memory-aware conversation
if user_input:
    # Technical Execution Flow for Streamlit Memory System:
    # 1. User enters question in Streamlit web interface
    # 2. Current question + Streamlit session history retrieved
    # 3. Complete conversation context assembled from session state
    # 4. Prompt template populated with system message + history + current question
    # 5. Enhanced prompt sent to LLM with full conversational context
    # 6. LLM generates contextually aware response
    # 7. Both question and response automatically saved to Streamlit session state
    # 8. Response displayed in web interface
    # 9. Session state persists across page interactions
    
    # Invoke memory-enhanced chain with web session management
    response = chain_with_history.invoke(
        # Current user input from Streamlit widget
        {"input": user_input},
        # Web session configuration
        {"configurable": {
            # session_id: Identifies the browser session
            # In production: could use user authentication, browser fingerprint, etc.
            "session_id": "streamlit_agile_session"
        }}
    )
    
    # Display AI response with enhanced formatting
    st.write("🤖 **Agile Coach:**")
    st.write(response.content)
    
    # Add visual separator between Q&A and history
    st.divider()

# =============================================================================
# CONVERSATION HISTORY VISUALIZATION
# =============================================================================

# Display conversation history section
st.subheader("📝 Conversation History")

# Check if there are messages in the session
if len(history_for_chain.messages) > 0:
    # Display each message in the conversation with proper formatting
    for i, message in enumerate(history_for_chain.messages):
        # Determine message role and apply appropriate styling
        if message.type == "human":
            # User messages with distinct styling
            st.write(f"**👤 You (Message {i//2 + 1}):**")
            st.write(f"> {message.content}")
        else:
            # AI messages with coach styling
            st.write(f"**🤖 Agile Coach:**")
            st.write(message.content)
        
        # Add spacing between message pairs
        if i < len(history_for_chain.messages) - 1:
            st.write("---")
    
    # Show conversation statistics
    st.write(f"💬 **Total Messages**: {len(history_for_chain.messages)} "
             f"({len(history_for_chain.messages)//2} exchanges)")
    
else:
    # Display when no conversation has started yet
    st.write("*No conversation history yet. Start by asking a question above!*")

# =============================================================================
# TECHNICAL INFORMATION AND DEBUGGING
# =============================================================================

# Add expandable section for technical details
with st.expander("🔧 Technical Details & Memory System Info"):
    st.markdown("""
    ### Memory System Architecture
    - **Storage**: Streamlit session state (survives page refresh)
    - **Isolation**: Each browser session has separate memory
    - **Persistence**: Memory lasts for the duration of the browser session
    - **Integration**: Native Streamlit session state management
    
    ### Conversation Flow
    1. User input captured via Streamlit widget
    2. Session history retrieved from `st.session_state`
    3. Complete context sent to OpenAI GPT-4o
    4. Response generated with full conversation awareness
    5. New messages automatically stored in session state
    
    ### Session Management
    - **Session ID**: `streamlit_agile_session`
    - **Memory Type**: StreamlitChatMessageHistory
    - **State Persistence**: Automatic via Streamlit framework
    """)
    
    # Show raw session state for debugging
    if st.checkbox("Show Raw Session State"):
        st.json(dict(st.session_state))

# Add sidebar with usage instructions
st.sidebar.markdown("""
### 🚀 How to Use

1. **Ask Questions**: Enter Agile-related questions
2. **Follow Up**: Reference previous answers naturally
3. **View History**: See the complete conversation below
4. **Memory Persists**: Refresh the page - memory remains!

### 💡 Try These Conversational Examples

**Initial Question:**
"What is a Sprint Review?"

**Follow-up Questions:**
- "How long should it be?"
- "Who should attend that meeting?"
- "What's the difference between that and a Sprint Retrospective?"

### 🔄 Running This App
```bash
streamlit run streamlit_chathistory_demo.py
```

### 🧠 Memory Features
- ✅ Remembers conversation context
- ✅ Handles follow-up questions
- ✅ Survives page refreshes
- ✅ Visual conversation history
- ✅ Session isolation per browser
""")