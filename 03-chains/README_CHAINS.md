# LangChain Chains: Complete Guide

## What are Chains?

**Chains** in LangChain are sequences of operations that connect different components (prompts, models, parsers, tools) to create complex AI workflows. They enable you to build sophisticated applications by combining simple, reusable components.

## Core Concept

Think of chains as a pipeline where:
- **Input** flows through multiple processing steps
- **Output** of one step becomes input for the next
- **Each component** has a specific responsibility

```
Input → Component1 → Component2 → Component3 → Final Output
```

## Types of Chains

### 1. **Simple Chain**
- Single prompt → model → output
- Most basic form of chain
- Example: Question → LLM → Answer

```python
simple_chain = prompt | llm | output_parser
```

### 2. **Sequential Chain**
- Multiple chains executed in sequence
- Output of first chain becomes input of second
- Used in our speech generator example

```python
chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm
sequential = chain1 | chain2
```

### 3. **Parallel Chain**
- Multiple chains executed simultaneously
- All chains receive the same input
- Results can be combined

```python
from langchain.chains import ParallelChain
parallel = ParallelChain(chains=[chain1, chain2, chain3])
```

### 4. **Conditional Chain**
- Execute different paths based on conditions
- Dynamic routing based on input content

```python
from langchain.chains import ConditionalChain
conditional = ConditionalChain(
    condition=lambda x: "question" in x.lower(),
    if_true=qa_chain,
    if_false=general_chain
)
```

### 5. **Router Chain**
- Routes inputs to different specialized chains
- Based on content classification

```python
from langchain.chains.router import MultiPromptChain
router = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains
)
```

### 6. **Map-Reduce Chain**
- Process multiple documents/inputs
- Combine results into final output

```python
from langchain.chains import MapReduceChain
map_reduce = MapReduceChain(
    map_chain=summarize_chain,
    reduce_chain=combine_chain
)
```

## What Chains Do

### 1. **Orchestrate Workflows**
- Connect multiple AI operations
- Manage data flow between components
- Handle complex business logic

### 2. **Transform Data**
- Convert outputs to inputs for next step
- Parse and format responses
- Extract specific information

### 3. **Create Modularity**
- Reusable components
- Easy testing and debugging
- Better maintainability

### 4. **Enable Complex Applications**
- Multi-step reasoning
- Context preservation
- Error handling

## Chain Components

### **Prompts**
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write about {topic}"
)
```

### **Models (LLMs)**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
```

### **Output Parsers**
```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
```

### **Custom Functions**
```python
def custom_processor(text):
    return text.upper()
```

## Chain Syntax

### **Pipe Operator (|)**
- Modern LangChain syntax
- Clean and readable
- Left-to-right execution

```python
chain = prompt | llm | parser
```

### **Traditional Syntax**
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

## Best Practices

### 1. **Keep Chains Simple**
- Single responsibility per chain
- Easy to test and debug

### 2. **Use Descriptive Names**
```python
title_generation_chain = title_prompt | llm | parser
speech_writing_chain = speech_prompt | llm
```

### 3. **Error Handling**
```python
try:
    result = chain.invoke(input_data)
except Exception as e:
    print(f"Chain execution failed: {e}")
```

### 4. **Input Validation**
```python
def validate_input(data):
    if not data.get("topic"):
        raise ValueError("Topic is required")
    return data
```

## Common Use Cases

### 1. **Content Generation**
- Topic → Title → Content → Formatting

### 2. **Question Answering**
- Question → Context Retrieval → Answer Generation

### 3. **Document Processing**
- Document → Chunking → Summarization → Insights

### 4. **Conversational AI**
- Input → Context → Response → Memory Update

## Speech Generator Chain Analysis

Our example demonstrates a **Sequential Chain**:

```
Topic Input → Title Generation → Speech Writing → Output Display
```

**Flow Breakdown:**
1. **User Input**: Topic (e.g., "Climate Change")
2. **First Chain**: Topic → Title Prompt → LLM → Title Parser → Display
3. **Second Chain**: Title → Speech Prompt → LLM → Full Speech
4. **Output**: Complete speech with title

**Technical Details:**
- Uses pipe operator for clean syntax
- Lambda function for dual purpose (display + pass-through)
- Sequential execution ensures title is available for speech

This pattern is perfect for multi-step content generation where each step builds upon the previous one.
