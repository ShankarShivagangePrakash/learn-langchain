import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

# 1. Define a tool using the @tool decorator and specify the tool name
@tool("get_current_timestamp")
def get_current_time(format_string: str = "%I:%M %p") -> str:
    """
    Returns the current time in the specified format. 
    Defaults to H:MM AM/PM format (e.g., "12:30 PM").
    """
    import datetime
    now = datetime.datetime.now()
    return now.strftime(format_string)

# List of tools available to the agent
tools = [get_current_time]

# 2. Create the LLM object (must be a model that supports tool calling, e.g., gpt-4.1)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=OPENAI_API_KEY)

# 3. Pull a prompt template from the LangChain hub (standard template for tool calling agents)
prompt = hub.pull("hwchase17/openai-tools-agent:c1867281")

# 4. Create the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Create an AgentExecutor to run the agent and manage the tool execution loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Invoke the agent with a query
response = agent_executor.invoke({"input": "What time is it right now?"})

print("Response:", response['output'])