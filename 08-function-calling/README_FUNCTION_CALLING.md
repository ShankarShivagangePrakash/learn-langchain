# Function Calling (Tool Calling) in LangChain

## What is Function / Tool Calling?

**Function calling** (also referred to as **tool calling**) is a capability of modern LLMs that allows the model to **request the execution of external functions/api** during a conversation. Instead of the LLM trying to answer everything from its training data alone, it can recognize when a user's question requires real-time or external data and **decide to call a specific function/api** to get that information.

### How It Works — Step by Step

```
User Query ──▶ LLM analyzes the query
                   │
                   ▼
           Does this need a tool?
              /         \
            No           Yes
             │             │
             ▼             ▼
     Direct answer    LLM outputs a structured
     from training    function call request
     data             (tool name + arguments)
                           │
                           ▼
                  AgentExecutor runs
                  the actual function
                           │
                           ▼
                  Function result is sent
                  back to the LLM
                           │
                           ▼
                  LLM forms a final
                  natural language response
```

1. **User asks a question** — e.g., *"What time is it right now?"*
2. **LLM analyzes the query** — It determines that answering this requires real-time data it doesn't have.
3. **LLM generates a tool call** — Instead of hallucinating an answer, the model outputs a structured JSON request specifying which function to call and with what arguments.
4. **The framework executes the function** — LangChain's `AgentExecutor` intercepts the tool call, runs the actual Python function, and captures the result.
5. **Result is fed back to the LLM** — The function's return value is sent back to the model as context.
6. **LLM generates the final response** — The model uses the function result to compose a natural language answer for the user.

> **Key Insight:** The LLM never executes code itself. It only *requests* that a function be called. The actual execution happens on your machine via the LangChain framework.

---

## Why is Tool Calling Important?

| Problem | How Tool Calling Solves It |
|---|---|
| LLMs have a **knowledge cutoff** and can't access real-time data | Tools can fetch live data (time, weather, stock prices, etc.) |
| LLMs can **hallucinate** answers for factual queries | Tools provide verified, deterministic results |
| LLMs can't **interact with external systems** (databases, APIs, file systems) | Tools bridge the gap between the LLM and the outside world |
| LLMs are poor at **precise computation** (math, dates, etc.) | Tools can perform exact calculations |

### Common Use Cases

- **Real-time information** — Current time, weather, news, stock prices
- **Database queries** — Look up customer records, inventory, order history
- **API integrations** — Send emails, create calendar events, post to Slack
- **Calculations** — Math operations, unit conversions, financial computations
- **File operations** — Read/write files, process documents, generate reports
- **Search** — Web search, knowledge base lookup, semantic search over documents

---

## Code Walkthrough — `tool_call_demo.py`

### Prerequisites

- Python 3.9+
- An **OpenAI API key** with access to a tool-calling model (e.g., `gpt-4.1`, `gpt-4o`, `gpt-3.5-turbo`)

### Key Components

#### 1. Defining a Tool with `@tool`

```python
from langchain_core.tools import tool

@tool("get_current_timestamp")
def get_current_time(format_string: str = "%I:%M %p") -> str:
    """
    Returns the current time in the specified format.
    Defaults to H:MM AM/PM format (e.g., "12:30 PM").
    """
    import datetime
    now = datetime.datetime.now()
    return now.strftime(format_string)
```

- The `@tool` decorator converts a regular Python function into a LangChain **Tool**.
- The **string argument** `"get_current_timestamp"` sets the tool's name — this is what the LLM sees and uses to identify the tool.
- The **docstring** is critical — the LLM reads it to understand *when* and *how* to use the tool.
- **Type hints** and **default values** on parameters help the LLM generate correct arguments.

#### 2. Creating the LLM

```python
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=OPENAI_API_KEY)
```

- The model **must support tool calling** (OpenAI's `gpt-4.1`, `gpt-4o`, `gpt-3.5-turbo` all support it).
- `temperature=0` ensures deterministic, consistent responses.

#### 3. Prompt Template from LangChain Hub

```python
prompt = hub.pull("hwchase17/openai-tools-agent:c1867281")
```

- Pulls a pre-built prompt template designed for OpenAI tool-calling agents.
- This prompt includes placeholders for the chat history, user input, and the agent scratchpad (where intermediate tool call results are stored).

#### 4. Creating the Agent

```python
agent = create_tool_calling_agent(llm, tools, prompt)
```

- Combines the LLM, tools list, and prompt into an **agent** — a reasoning loop that can decide when to call tools.

#### 5. AgentExecutor

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

- The `AgentExecutor` manages the **execution loop**: it sends the query to the LLM, checks if the LLM wants to call a tool, runs the tool, sends the result back, and repeats until the LLM produces a final answer.
- `verbose=True` prints the full chain of thought and tool calls to the console — useful for debugging and learning.

#### 6. Invoking the Agent

```python
response = agent_executor.invoke({"input": "What time is it right now?"})
print("Response:", response['output'])
```

- The agent receives the user query, determines that it needs the `get_current_timestamp` tool, calls it, and returns the result in natural language.

---

## How to Run

### 1. Set up the environment

```bash
# Navigate to the project root
cd learn-langchain

# Create and activate a virtual environment (if not already done)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the demo

```bash
cd 08-function-calling
python tool_call_demo.py
```

### Expected Output

```
> Entering new AgentExecutor chain...

Invoking: `get_current_timestamp` with `{'format_string': '%I:%M %p'}`

12:30 PM

The current time is 12:30 PM.

> Finished chain.
Response: The current time is 12:30 PM.
```

---

## Tool Calling vs. Function Calling — Terminology

| Term | Meaning |
|---|---|
| **Function Calling** | The original term used by OpenAI for this capability |
| **Tool Calling** | The standardized term used across LangChain (works with OpenAI, Anthropic, Google, etc.) |

LangChain uses **"tool calling"** as the unified abstraction. Under the hood, when using OpenAI models, it maps to OpenAI's function calling API. This means the same LangChain code works across different LLM providers that support tool use.

---

## Key Concepts Summary

| Concept | Description |
|---|---|
| `@tool` decorator | Converts a Python function into a LangChain Tool with name, description, and schema |
| `ChatOpenAI` | LangChain wrapper for OpenAI chat models |
| `create_tool_calling_agent` | Creates an agent that can decide when to invoke tools |
| `AgentExecutor` | Runs the agent loop — handles tool invocations and feeds results back to the LLM |
| `hub.pull()` | Fetches a reusable prompt template from the LangChain Hub |
| `verbose=True` | Enables detailed logging of the agent's reasoning and tool calls |

---

## Tips for Writing Good Tools

1. **Clear docstrings** — The LLM uses the docstring to decide when to call the tool. Be descriptive.
2. **Descriptive parameter names** — Use meaningful names with type hints so the LLM knows what arguments to pass.
3. **Sensible defaults** — Provide default values for optional parameters.
4. **Specific tool names** — The `@tool("name")` argument helps the LLM distinguish between multiple tools.
5. **Keep tools focused** — Each tool should do one thing well. Compose multiple tools for complex workflows.
