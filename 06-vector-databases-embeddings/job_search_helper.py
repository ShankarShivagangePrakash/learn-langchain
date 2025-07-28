"""
Semantic Job Search Helper - Vector-Based Job Matching System
============================================================

This script demonstrates how to build a semantic job search system using vector embeddings
and similarity search. Instead of keyword-based matching, it finds jobs based on semantic
similarity between user queries and job descriptions, enabling more intelligent job matching.

Key Features:
- Semantic Search: Find jobs by meaning, not just keywords
- Vector Embeddings: Convert job descriptions to mathematical representations
- Similarity Matching: Rank jobs by relevance to user queries
- Flexible Queries: Natural language job search capabilities
- Document Chunking: Handle long job descriptions efficiently

How It Works:
1. Load job listings from text file
2. Split job descriptions into manageable chunks
3. Convert chunks to vector embeddings using OpenAI
4. Store embeddings in Chroma vector database
5. Process user queries through semantic similarity search
6. Return most relevant job matches

Technical Architecture:
- Document Processing: TextLoader + RecursiveCharacterTextSplitter
- Embeddings: OpenAI text-embedding-ada-002 (1536 dimensions)
- Vector Storage: Chroma database for similarity search
- Retrieval: Cosine similarity-based ranking
- Output: Ranked list of relevant job chunks

Use Cases:
- Job seekers finding relevant positions
- Recruiters matching candidates to roles
- Career guidance and job recommendation systems
- Skills-based job filtering and discovery

Example Queries:
- "Python developer with machine learning experience"
- "Remote marketing manager position"
- "Entry level data analyst jobs"
- "Senior frontend engineer React"

Author: Learn LangChain Course
Date: July 2025
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# =============================================================================
# CONFIGURATION & EMBEDDING MODEL SETUP
# =============================================================================

# Retrieve OpenAI API key for embedding generation
# Required for converting job descriptions and queries into vector representations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings model for semantic understanding
# Technical Details:
# - Model: text-embedding-ada-002 (optimized for search and similarity)
# - Dimensions: 1536 (high-dimensional semantic representation)
# - Purpose: Convert job descriptions and queries into comparable vectors
# Note: Variable name 'llm' is misleading - this is an embedding model, not a language model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# =============================================================================
# JOB LISTINGS DATA PROCESSING PIPELINE
# =============================================================================

# Load job listings from text file
# Expected format: Text file containing job descriptions, one per section
# The file should contain structured job information including:
# - Job titles, companies, requirements, descriptions, etc.
# Technical Note: TextLoader reads the entire file and creates a single document
current_dir = os.path.dirname(os.path.abspath(__file__))
job_file_path = os.path.join(current_dir, "job_listings.txt")
document = TextLoader(job_file_path).load()

# Split job document into smaller, searchable chunks
# Chunking Strategy for Job Search:
# - chunk_size=200: Small chunks ensure specific job details are preserved
# - chunk_overlap=10: Minimal overlap to avoid redundancy while maintaining context
# - Purpose: Each chunk represents a focused piece of job information
# - Benefits: More precise matching, better relevance scoring
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,         # Smaller chunks for precise job matching
    chunk_overlap=10        # Minimal overlap for job-specific content
)
chunks = text_splitter.split_documents(document)

print(f"📄 Loaded job listings and split into {len(chunks)} searchable chunks")
print(f"🔍 Each chunk contains ~200 characters of job information")
# =============================================================================
# VECTOR DATABASE CREATION & INDEXING
# =============================================================================

# Create Chroma vector database from job listing chunks
# Technical Process:
# 1. Convert each job chunk to 1536-dimensional embedding vector
# 2. Store vectors with metadata (source document, chunk index, content)
# 3. Build searchable index for fast similarity retrieval
# 4. Enable semantic search capabilities across all job listings
print(f"🔄 Creating vector database and generating embeddings...")
print(f"⏳ This may take a moment - converting {len(chunks)} chunks to vectors...")

# Chroma.from_documents() performs batch embedding generation and storage
# - chunks: List of document chunks containing job information
# - embeddings_model: OpenAI model for vector generation
# - Result: Searchable vector database with job listings
vector_db = Chroma.from_documents(chunks, embeddings_model)

# Create retriever interface for semantic job search
# Retriever configuration:
# - Default: Returns top 4 most similar chunks
# - Similarity metric: Cosine similarity (measures angle between vectors)
# - Search method: Approximate nearest neighbor for fast results
job_retriever = vector_db.as_retriever()

print(f"✅ Vector database created successfully!")
print(f"🎯 Ready for semantic job search queries")

# =============================================================================
# INTERACTIVE SEMANTIC JOB SEARCH INTERFACE
# =============================================================================

print(f"\n🔍 Semantic Job Search Helper")
print(f"Find jobs using natural language descriptions!")
print(f"Example queries:")
print(f"  • 'Python developer with machine learning'")
print(f"  • 'Remote marketing manager'")
print(f"  • 'Entry level data scientist'")
print(f"  • 'Senior React frontend engineer'")
print(f"  • 'Product manager with AI experience'")
print()

# Get user's job search query
user_query = input("🎯 Enter your job search query: ").strip()

if user_query:
    # Process query through semantic search system
    # Technical Execution Flow:
    # 1. Convert user query to 1536-dimensional embedding vector
    # 2. Calculate cosine similarity with all job listing vectors
    # 3. Rank job chunks by similarity score (0-1 scale)
    # 4. Return top matching chunks (default: 4 results)
    # 5. Present results ordered by relevance
    
    print(f"\n🔄 Searching for jobs similar to: '{user_query}'")
    print(f"📊 Using semantic similarity matching...")
    
    # Invoke semantic search retrieval
    # invoke() method:
    # - Converts query to embedding automatically
    # - Performs similarity search in vector space
    # - Returns ranked list of most relevant job chunks
    relevant_jobs = job_retriever.invoke(user_query)
    
    # =============================================================================
    # SEARCH RESULTS DISPLAY & ANALYSIS
    # =============================================================================
    
    print(f"\n✅ Found {len(relevant_jobs)} relevant job matches:")
    print(f"📈 Results ranked by semantic similarity")
    print("=" * 60)
    
    # Display each relevant job chunk with formatting
    for i, job_doc in enumerate(relevant_jobs, 1):
        print(f"\n🎯 **Match #{i}** (Relevance: High to Low)")
        print("-" * 40)
        
        # Display job content with proper formatting
        job_content = job_doc.page_content.strip()
        print(f"📋 {job_content}")
        
        # Show metadata if available (source document, chunk info, etc.)
        if hasattr(job_doc, 'metadata') and job_doc.metadata:
            metadata = job_doc.metadata
            if 'source' in metadata:
                print(f"📁 Source: {metadata['source']}")
        
        # Add separator between results
        if i < len(relevant_jobs):
            print("─" * 40)
    
    # =============================================================================
    # SEARCH INSIGHTS & EDUCATIONAL INFORMATION
    # =============================================================================
    
    print(f"\n💡 **How This Search Worked:**")
    print(f"   1. Your query was converted to a 1536-dimensional vector")
    print(f"   2. System compared your vector with all job listing vectors")
    print(f"   3. Jobs with similar meanings ranked highest (cosine similarity)")
    print(f"   4. Results show jobs semantically related to your query")
    
    print(f"\n🎯 **Why Semantic Search is Better:**")
    print(f"   • Finds jobs by meaning, not just exact keyword matches")
    print(f"   • 'ML engineer' matches 'machine learning developer'")
    print(f"   • 'Remote work' matches 'work from home', 'telecommute'")
    print(f"   • Understands synonyms and related concepts automatically")
    
    print(f"\n🚀 **Try More Searches:**")
    print(f"   • Different skill combinations")
    print(f"   • Various experience levels")
    print(f"   • Specific technologies or tools")
    print(f"   • Industry-specific terms")
    
    # Optional: Show technical details about the search
    show_technical = input(f"\n❓ Show technical search details? (y/n): ").lower().strip()
    if show_technical == 'y':
        print(f"\n🔧 **Technical Search Details:**")
        print(f"   • Embedding Model: text-embedding-ada-002")
        print(f"   • Vector Dimensions: 1536")
        print(f"   • Similarity Metric: Cosine similarity")
        print(f"   • Search Method: Approximate nearest neighbor")
        print(f"   • Total Job Chunks: {len(chunks)}")
        print(f"   • Results Returned: {len(relevant_jobs)}")
        print(f"   • Query Processing: Automatic embedding generation")

else:
    print("❌ No query provided. Please enter a job search query.")
    print("💡 Try queries like 'Python developer' or 'marketing manager'")

# =============================================================================
# EDUCATIONAL EXAMPLES & USE CASES
# =============================================================================

print(f"\n📚 **Understanding Semantic Job Search:**")
print(f"""
🔍 SEMANTIC MATCHING EXAMPLES:
Query: "Python developer"
Matches: "Python programmer", "software engineer Python", "backend developer Python"

Query: "Remote work"  
Matches: "work from home", "telecommute", "distributed team", "remote position"

Query: "Machine learning"
Matches: "ML engineer", "AI developer", "data scientist", "deep learning"

⚡ ADVANTAGES OVER KEYWORD SEARCH:
• Understands synonyms and related terms
• Finds relevant jobs even with different wording
• Captures semantic relationships between concepts
• Reduces need for exact keyword matching
• Improves job discovery and matching accuracy

🛠️ TECHNICAL IMPLEMENTATION:
• Vector embeddings capture semantic meaning
• Similarity search finds conceptually related content
• Ranking by relevance score ensures best matches first
• Chunk-based processing handles long job descriptions
• Real-time query processing for interactive search
""")