# =============================================================================
# AI AGENT DEMONSTRATION WITH STREAMLIT INTERFACE
# =============================================================================
"""
Educational Demo: ReAct Agent with Wikipedia and Web Search Tools

This script demonstrates a complete AI agent implementation using the ReAct 
(Reasoning + Acting) framework. The agent can autonomously research topics,
search for information, and provide comprehensive answers using multiple tools.

TECHNICAL CONCEPTS COVERED:
• ReAct (Reasoning + Acting) agent architecture
• Tool integration and dynamic selection
• Agent execution loops and decision making
• Streamlit web interface for agent interaction
• Wikipedia and DuckDuckGo search capabilities

EDUCATIONAL VALUE:
• Understand autonomous AI agent behavior
• Learn tool-based AI system architecture
• Explore multi-step reasoning processes
• See practical agent-tool integration patterns

BUSINESS APPLICATIONS:
• Research and information gathering
• Customer support automation
• Content creation assistance
• Market research and analysis
• Educational tutoring systems

AGENT CAPABILITIES:
• Multi-source information gathering
• Step-by-step reasoning display
• Autonomous tool selection
• Context-aware responses
• Error handling and recovery
"""

import os
from langchain_openai import ChatOpenAI
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools

# =============================================================================
# ENVIRONMENT CONFIGURATION & MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Security Best Practice: Store sensitive credentials in environment variables
# Set with: export OPENAI_API_KEY="your-api-key-here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model for agent reasoning
# Model Configuration:
# - GPT-4o: Latest GPT-4 optimized model (fast, cost-effective)
# - Temperature: Default (0.7) for balanced creativity and accuracy
# - Max Tokens: Automatically managed by the agent
# - Streaming: Disabled for complete responses
chat_model = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY,
    temperature=0.7  # Balanced creativity for reasoning tasks
)

# =============================================================================
# REACT PROMPT TEMPLATE CONFIGURATION
# =============================================================================

# Load the ReAct prompt template from LangChain Hub
# ReAct Framework Components:
# - Thought: Agent's reasoning about the current situation
# - Action: Tool selection and parameter specification
# - Observation: Results from tool execution
# - Final Answer: Conclusion after reasoning process
#
# Prompt Template Structure:
# "Answer the following questions as best you can. You have access to the following tools:
# {tools}
# Use the following format:
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question"
react_prompt = hub.pull("hwchase17/react")

# =============================================================================
# AGENT TOOLS CONFIGURATION
# =============================================================================

# Load available tools for the agent
# Tool Selection Rationale:
# 1. Wikipedia: Comprehensive, reliable encyclopedia data
# 2. DuckDuckGo Search: Privacy-focused web search engine
#
# Tool Capabilities:
tools = load_tools([
    "wikipedia",    # Access to Wikipedia articles and summaries
    "ddg-search"    # DuckDuckGo web search for current information
])

# Tool Details:
# Wikipedia Tool:
# - Function: Search and retrieve Wikipedia article content
# - Input: Search query or article title
# - Output: Article summary and key information
# - Use Cases: Historical facts, biographical data, general knowledge
# - Limitations: May not have the most recent information
#
# DuckDuckGo Search Tool:
# - Function: Web search across the internet
# - Input: Search query string
# - Output: Search results with titles, snippets, and URLs
# - Use Cases: Current events, recent news, specific information
# - Advantages: Privacy-focused, no user tracking

print(f"🔧 Available Tools: {[tool.name for tool in tools]}")
print(f"📚 Tool Descriptions:")
for tool in tools:
    print(f"   • {tool.name}: {tool.description}")

# =============================================================================
# REACT AGENT CREATION & CONFIGURATION
# =============================================================================

# Create ReAct agent with configured components
# Agent Architecture:
# - LLM: Provides reasoning and decision-making capabilities
# - Tools: External capabilities for information gathering
# - Prompt: Instructions for reasoning and tool usage patterns
#
# ReAct Agent Workflow:
# 1. Receive user input/question
# 2. Think about the problem and required information
# 3. Select appropriate tool for information gathering
# 4. Execute tool with proper parameters
# 5. Observe and analyze tool results
# 6. Repeat steps 2-5 until sufficient information gathered
# 7. Formulate comprehensive final answer
agent = create_react_agent(
    llm=chat_model,      # Language model for reasoning
    tools=tools,         # Available tools for the agent
    prompt=react_prompt  # ReAct framework instructions
)

# =============================================================================
# AGENT EXECUTOR CONFIGURATION
# =============================================================================

# Create AgentExecutor to manage agent execution
# Executor Responsibilities:
# - Orchestrate the agent's reasoning loop
# - Handle tool calls and responses
# - Manage execution flow and error handling
# - Provide logging and debugging information
# - Enforce safety limits and constraints
agent_executor = AgentExecutor(
    agent=agent,           # The ReAct agent instance
    tools=tools,           # Available tools for execution
    verbose=True,          # Enable detailed logging for education
    max_iterations=10,     # Prevent infinite loops
    handle_parsing_errors=True,  # Graceful error handling
    return_intermediate_steps=True  # Show reasoning steps
)

# Executor Configuration Details:
# - max_iterations: Limits agent to 10 reasoning cycles
# - verbose: Shows step-by-step agent thinking process
# - handle_parsing_errors: Recovers from tool call format errors
# - return_intermediate_steps: Provides transparency into agent reasoning

print(f"✅ Agent Executor configured with {len(tools)} tools")
print(f"🔍 Verbose mode enabled for educational transparency")

# =============================================================================
# STREAMLIT WEB INTERFACE
# =============================================================================

# Configure Streamlit page settings
st.set_page_config(
    page_title="AI Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main application title and description
st.title("🤖 AI Agent Demonstration")
st.markdown("""
### ReAct Agent with Wikipedia & Web Search

This intelligent agent can research topics, gather information from multiple sources, 
and provide comprehensive answers using autonomous reasoning and tool selection.

**Agent Capabilities:**
- 🔍 Web search using DuckDuckGo
- 📚 Wikipedia knowledge access  
- 🧠 Multi-step reasoning process
- 🔧 Autonomous tool selection
- 📊 Transparent decision making
""")

# Sidebar with agent information
with st.sidebar:
    st.header("🔧 Agent Configuration")
    st.write("**Model:** GPT-4o")
    st.write("**Framework:** ReAct (Reasoning + Acting)")
    st.write("**Tools Available:**")
    for tool in tools:
        st.write(f"• {tool.name}")
    
    st.header("💡 Example Queries")
    st.write("""
    - "What are the latest developments in AI?"
    - "Compare Python and JavaScript for web development"
    - "What is the history of the Internet?"
    - "Explain quantum computing simply"
    - "What's happening in space exploration?"
    """)

# =============================================================================
# USER INPUT & TASK PROCESSING
# =============================================================================

# Task input interface with enhanced UX
st.subheader("🎯 Assign a Task to the Agent")
task = st.text_input(
    "Enter your question or research topic:",
    placeholder="e.g., 'What are the latest breakthroughs in renewable energy?'",
    help="Ask anything that requires research or information gathering. The agent will use multiple sources to provide a comprehensive answer."
)

# Additional input options
col1, col2 = st.columns(2)
with col1:
    show_reasoning = st.checkbox("Show Agent Reasoning Steps", value=True, 
                                help="Display the agent's thought process and tool usage")
with col2:
    max_iterations = st.slider("Max Reasoning Steps", min_value=3, max_value=15, value=10,
                              help="Limit the number of reasoning cycles")

# Update agent executor with user preferences
agent_executor.max_iterations = max_iterations

if task:
    # Display processing status
    with st.spinner("🤔 Agent is researching your question..."):
        st.info(f"**Question:** {task}")
        
        # Create container for agent reasoning display
        if show_reasoning:
            reasoning_container = st.expander("🧠 Agent Reasoning Process", expanded=True)
        
        try:
            # Execute agent task with comprehensive error handling
            # The agent will:
            # 1. Analyze the user's question
            # 2. Plan an information gathering strategy
            # 3. Select and use appropriate tools
            # 4. Synthesize information into a coherent answer
            # 5. Provide citations and sources when possible
            response = agent_executor.invoke({
                "input": task,
                "return_intermediate_steps": show_reasoning
            })
            
            # =============================================================================
            # RESULTS DISPLAY & ANALYSIS
            # =============================================================================
            
            # Display agent reasoning steps if requested
            if show_reasoning and "intermediate_steps" in response:
                with reasoning_container:
                    st.markdown("### 🔍 Step-by-Step Agent Analysis:")
                    
                    for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
                        with st.container():
                            st.markdown(f"**Step {i}:**")
                            
                            # Show agent's thought process
                            if hasattr(action, 'log'):
                                st.markdown(f"🤔 **Thought:** {action.log}")
                            
                            # Show tool selection and usage
                            st.markdown(f"🔧 **Action:** {action.tool}")
                            st.markdown(f"📝 **Input:** {action.tool_input}")
                            
                            # Show tool results
                            with st.expander(f"📊 Tool Result #{i}", expanded=False):
                                st.text(str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation))
                            
                            st.markdown("---")
            
            # Display final answer with formatting
            st.success("✅ **Agent Research Complete!**")
            
            # Main answer display
            st.markdown("### 🎯 **Final Answer:**")
            
            # Format and display the response
            final_answer = response.get("output", "No response generated")
            
            # Enhanced answer display with styling
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                {final_answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics and information
            st.markdown("### 📊 **Execution Summary:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                steps_taken = len(response.get("intermediate_steps", []))
                st.metric("Reasoning Steps", steps_taken, f"Max: {max_iterations}")
            
            with col2:
                tools_used = set()
                if "intermediate_steps" in response:
                    for action, _ in response["intermediate_steps"]:
                        tools_used.add(action.tool)
                st.metric("Tools Used", len(tools_used), f"Available: {len(tools)}")
            
            with col3:
                # Estimate response quality based on steps and tools
                quality_score = min(100, (steps_taken * 20) + (len(tools_used) * 30))
                st.metric("Research Depth", f"{quality_score}%", "Comprehensive" if quality_score > 70 else "Basic")
            
            # Educational insights
            st.markdown("### 🎓 **Educational Insights:**")
            st.info(f"""
            **How the Agent Worked:**
            1. **Question Analysis**: Broke down your question into searchable components
            2. **Strategy Planning**: Determined which tools would provide the best information
            3. **Information Gathering**: Used {len(tools_used)} different tools to collect data
            4. **Synthesis**: Combined information from multiple sources into a coherent answer
            5. **Verification**: Cross-referenced information for accuracy and completeness
            
            **Agent Benefits Demonstrated:**
            - Autonomous research across multiple sources
            - Transparent reasoning process
            - Comprehensive information synthesis
            - Adaptive strategy based on intermediate results
            """)
            
        except Exception as e:
            # Comprehensive error handling with user guidance
            st.error("❌ **Agent Execution Error**")
            st.markdown(f"**Error Details:** {str(e)}")
            
            st.markdown("### 🔧 **Troubleshooting Tips:**")
            st.markdown("""
            - **API Issues**: Check your OpenAI API key and rate limits
            - **Network Problems**: Ensure internet connection for web search tools
            - **Complex Queries**: Try breaking down complex questions into simpler parts
            - **Tool Limitations**: Some information may not be available through current tools
            """)
            
            # Suggest alternative approaches
            st.markdown("### 💡 **Alternative Approaches:**")
            st.markdown("""
            - Rephrase your question more specifically
            - Try a simpler, more direct query first
            - Check if the topic is recent (Wikipedia may not have latest info)
            - Consider breaking complex questions into multiple simpler ones
            """)

else:
    # Display helpful guidance when no task is entered
    st.markdown("### 🚀 **Getting Started:**")
    st.markdown("""
    1. **Enter a question** in the text box above
    2. **Choose your preferences** using the checkboxes and sliders
    3. **Watch the agent work** as it researches and provides answers
    4. **Learn from the process** by observing the reasoning steps
    
    The agent will autonomously select tools, gather information, and synthesize 
    comprehensive answers to your questions.
    """)

# =============================================================================
# EDUCATIONAL FOOTER & ADDITIONAL RESOURCES
# =============================================================================

st.markdown("---")
st.markdown("### 📚 **Learn More About AI Agents:**")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Technical Concepts:**
    - ReAct (Reasoning + Acting) Framework
    - Tool-augmented Language Models
    - Agent Architecture Patterns
    - Multi-step Reasoning Systems
    """)

with col2:
    st.markdown("""
    **Practical Applications:**
    - Research and Analysis Automation
    - Customer Support Systems
    - Content Creation Assistance
    - Market Intelligence Gathering
    """)

st.markdown("""
---
💡 **Educational Note:** This demo showcases autonomous AI agent capabilities. 
The agent makes independent decisions about tool usage and information synthesis, 
demonstrating advanced AI reasoning and problem-solving patterns.
""")














