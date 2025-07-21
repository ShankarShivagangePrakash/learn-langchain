"""
PDF RAG (Retrieval-Augmented Generation) Demo
=============================================

This script demonstrates a RAG implementation specifically designed for PDF documents.
It extends the basic RAG concept to handle PDF files, making it ideal for academic papers,
research documents, reports, and other PDF-based knowledge sources.

Key Differences from Basic RAG:
    - Uses PyPDFLoader instead of TextLoader for PDF parsing
    - Handles multi-page documents with metadata preservation
    - Optimized for academic/research content structure
    - Maintains page references for source attribution

Technical Features:
    - PDF text extraction with page-level granularity
    - Semantic chunking with context preservation
    - Vector-based similarity search
    - LLM-powered response generation with source grounding

Use Cases:
    - Academic research assistance
    - Legal document analysis
    - Technical documentation Q&A
    - Report summarization and querying
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

"""
=============================================================================
PDF DOCUMENT PROCESSING PIPELINE
=============================================================================

    Step 1: Load PDF document using PyPDFLoader
    PyPDFLoader advantages over basic TextLoader:
    - Handles multi-page PDFs automatically
    - Preserves page metadata (page numbers, document structure)
    - Deals with PDF-specific formatting and encoding issues
    - Extracts text while maintaining readability
    Technical Note: PyPDFLoader uses pypdf library under the hood for PDF parsing
"""
document = PyPDFLoader("academic_research_data.pdf").load()

"""
    Step 2: Split PDF content into manageable chunks
    Critical for PDF processing due to:
    - Academic papers often have long, dense content
    - Need to preserve logical document structure
    - Balance between context preservation and processing efficiency
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Optimal size for academic content (abstracts, paragraphs)
    chunk_overlap=200       # Overlap ensures continuity across chunk boundaries
)
chunks = text_splitter.split_documents(document)

"""
    Step 3: Create vector store with PDF-specific considerations
    Process:
    1. Each text chunk is converted to embeddings via OpenAI API
    2. Embeddings capture semantic meaning of academic/research content
    3. Chroma stores vectors with metadata (page numbers, source info)
    4. Creates searchable index for similarity-based retrieval
"""
vector_store = Chroma.from_documents(chunks, embeddings)

""""
    Step 4: Initialize retriever for semantic search
    Retriever configuration:
    - Uses cosine similarity for finding relevant chunks
    - Returns most semantically similar content to user queries
    - Maintains page-level attribution for source verification
"""
retriever = vector_store.as_retriever()

# =============================================================================
# PROMPT ENGINEERING FOR ACADEMIC/RESEARCH CONTENT
# =============================================================================

# Define specialized prompt template for PDF/academic content
# Optimized for research documents, papers, and technical content
prompt_template = ChatPromptTemplate.from_messages([
    # System prompt: Tailored for academic/research assistance
    ("system", """You are an assistant for answering questions.
    Use the provided context to respond. If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    # Human message placeholder for user questions
    # Supports queries about research findings, methodologies, data, conclusions, etc.
    ("human", "{input}")
])

"""
    Create document chain for combining retrieved PDF content with LLM
    "stuff" strategy: All relevant PDF chunks are included in the prompt context
    Alternative strategies (map-reduce, refine) available for very large documents
    Technical flow: Retrieved chunks → Formatted context → LLM prompt → Generated response
"""
qa_chain = create_stuff_documents_chain(llm, prompt_template)

"""
    Create complete RAG chain for PDF-based Q&A
    Integration points:
    1. User query processing
    2. Vector similarity search in PDF content
    3. Context retrieval with page attribution
    4. LLM response generation
    5. Source-grounded answer delivery
"""
rag_chain = create_retrieval_chain(retriever, qa_chain)

# =============================================================================
# INTERACTIVE PDF Q&A INTERFACE
# =============================================================================

# Initialize PDF-based question-answering session
print("Chat with PDF Document")
print("Ask questions about the academic research data. Examples:")
print("- What is the main research question?")
print("- What methodology was used?")
print("- What are the key findings?")
print("- What data sources were analyzed?")
print()

# Capture user query about the PDF content
question = input("Your Question: ")

# Process question through PDF RAG pipeline
if question:
    """
        Technical execution flow:
        1. Query Embedding: User question → vector representation
        2. Similarity Search: Compare query vector with PDF chunk vectors
        3. Retrieval: Select most relevant PDF chunks (with page metadata)
        4. Context Assembly: Combine retrieved chunks into coherent context
        5. Prompt Construction: Insert context and question into template
        6. LLM Generation: Generate response grounded in PDF content
        7. Response Delivery: Return answer with implicit source attribution
    """
    response = rag_chain.invoke({"input": question})
    print(f"\nAnswer: {response['answer']}")