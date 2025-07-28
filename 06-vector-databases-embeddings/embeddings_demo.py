"""
OpenAI Embeddings Demo - Text to Vector Conversion
==================================================

This script demonstrates how to convert text into high-dimensional vector representations
using OpenAI's embedding models. Embeddings are numerical representations that capture
semantic meaning, enabling similarity comparisons and semantic search capabilities.

Key Concepts Demonstrated:
- Text-to-Vector Conversion: Transform human-readable text into mathematical vectors
- Semantic Representation: Capture meaning beyond keywords in numerical form
- High-Dimensional Vectors: Generate 1536-dimensional representations (OpenAI ada-002)
- Embedding API Usage: Practical implementation of embedding generation

What are Embeddings?
- Numerical representations of text in high-dimensional space
- Each dimension captures different aspects of meaning and context
- Similar texts produce similar vectors (close in vector space)
- Enable mathematical operations on semantic content

Technical Specifications:
- Model: text-embedding-ada-002 (OpenAI's latest embedding model)
- Dimensions: 1536 (each text produces a 1536-element vector)
- Input: Any text string (up to token limits)
- Output: List of floating-point numbers representing semantic content

Use Cases:
- Semantic Search: Find similar content based on meaning
- Content Clustering: Group related documents automatically
- Recommendation Systems: Suggest similar items based on vector similarity
- RAG Systems: Retrieve relevant documents for question answering
- Duplicate Detection: Identify similar or duplicate content

Mathematical Foundation:
- Cosine Similarity: Measure angle between vectors (0-1 scale)
- Euclidean Distance: Measure straight-line distance between points
- Dot Product: Measure vector alignment and magnitude

Author: Learn LangChain Course
Date: July 2025
"""

import os
from langchain_openai import OpenAIEmbeddings

# =============================================================================
# CONFIGURATION & EMBEDDING MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Required for accessing OpenAI's embedding services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings model
# Technical Details:
# - Model: text-embedding-ada-002 (default, most capable)
# - Dimensions: 1536 (fixed output size)
# - Max tokens: 8,191 tokens per input
# - Languages: Supports 100+ languages with varying quality
# - Cost: Optimized for cost-effectiveness vs performance
# Note: Variable name 'llm' is misleading - this is an embedding model, not a language model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# =============================================================================
# INTERACTIVE EMBEDDING DEMONSTRATION
# =============================================================================

# Get user input for text-to-vector conversion
print("🔢 OpenAI Embeddings Demo")
print("Convert any text into a 1536-dimensional vector representation!")
print("Examples to try:")
print("  - 'The cat sat on the mat'")
print("  - 'Machine learning and artificial intelligence'")
print("  - 'I love programming in Python'")
print()

# Capture user input
text = input("Enter the text to convert to embeddings: ").strip()

if text:
    # Process text through OpenAI's embedding model
    # Technical Process:
    # 1. Text tokenization: Convert text to tokens (sub-word units)
    # 2. Token encoding: Map tokens to numerical IDs
    # 3. Model processing: Pass through transformer layers
    # 4. Vector generation: Produce 1536-dimensional output vector
    # 5. Normalization: Ensure consistent vector magnitude
    
    print(f"\n🔄 Converting text to embeddings...")
    print(f"Input text: '{text}'")
    print(f"Text length: {len(text)} characters")
    
    # Generate embeddings using OpenAI's API
    # embed_query() is optimized for single query processing
    # Alternative: embed_documents() for batch processing multiple texts
    response = embeddings_model.embed_query(text)
    
    # =============================================================================
    # EMBEDDING ANALYSIS AND VISUALIZATION
    # =============================================================================
    
    # Display embedding characteristics
    print(f"\n✅ Embedding generated successfully!")
    print(f"📊 Vector dimensions: {len(response)}")
    print(f"📈 Vector type: List of floating-point numbers")
    
    # Show first few dimensions for illustration
    print(f"\n🔍 First 10 dimensions:")
    for i, value in enumerate(response[:10]):
        print(f"  Dimension {i+1}: {value:.6f}")
    
    # Show last few dimensions
    print(f"\n🔍 Last 5 dimensions:")
    for i, value in enumerate(response[-5:], len(response)-4):
        print(f"  Dimension {i}: {value:.6f}")
    
    # Calculate vector statistics
    import statistics
    import math
    
    vector_mean = statistics.mean(response)
    vector_std = statistics.stdev(response)
    vector_magnitude = math.sqrt(sum(x*x for x in response))
    
    print(f"\n📈 Vector Statistics:")
    print(f"  Mean value: {vector_mean:.6f}")
    print(f"  Standard deviation: {vector_std:.6f}")
    print(f"  Vector magnitude (L2 norm): {vector_magnitude:.6f}")
    
    # Demonstrate practical applications
    print(f"\n🎯 What can you do with this embedding?")
    print(f"  • Store in vector database for semantic search")
    print(f"  • Compare with other embeddings using cosine similarity")
    print(f"  • Use for content clustering and classification")
    print(f"  • Build recommendation systems")
    print(f"  • Implement RAG (Retrieval-Augmented Generation)")
    
    # Educational note about similarity
    print(f"\n💡 Key Insight:")
    print(f"  Texts with similar meanings will have similar vectors.")
    print(f"  For example: 'dog' and 'puppy' will have closer vectors")
    print(f"  than 'dog' and 'computer'.")
    
    # Show the complete vector (optional, can be very long)
    show_full_vector = input(f"\n❓ Show complete {len(response)}-dimensional vector? (y/n): ").lower().strip()
    if show_full_vector == 'y':
        print(f"\n📋 Complete embedding vector:")
        print(response)
    
    # Suggest next steps
    print(f"\n🚀 Next steps to explore:")
    print(f"  1. Generate embeddings for different texts")
    print(f"  2. Calculate similarity between embeddings")
    print(f"  3. Store embeddings in a vector database")
    print(f"  4. Build a semantic search system")
    
else:
    print("❌ No text provided. Please enter some text to convert to embeddings.")

# =============================================================================
# EDUCATIONAL EXAMPLES AND USE CASES
# =============================================================================

print(f"\n📚 Understanding Embeddings:")
print(f"""
🔤 TEXT TO VECTOR CONVERSION EXAMPLE:
Input: "The quick brown fox"
Output: [0.123, -0.456, 0.789, ..., 0.321] (1536 numbers)

🎯 SIMILARITY EXAMPLES:
High Similarity (vectors will be close):
  • "dog" ↔ "puppy" ↔ "canine"
  • "car" ↔ "automobile" ↔ "vehicle"
  • "happy" ↔ "joyful" ↔ "delighted"

Low Similarity (vectors will be distant):
  • "dog" ↔ "mathematics"
  • "car" ↔ "happiness" 
  • "music" ↔ "geology"

⚙️ TECHNICAL SPECIFICATIONS:
  • Model: text-embedding-ada-002
  • Dimensions: 1536
  • Input limit: 8,191 tokens
  • Output: Normalized floating-point vector
  • Similarity metric: Cosine similarity (recommended)

🛠️ PRACTICAL APPLICATIONS:
  • Semantic Search: Find documents by meaning, not keywords
  • Content Clustering: Group similar articles automatically
  • Recommendation: "Users who liked X also liked Y"
  • Question Answering: Retrieve relevant context for LLMs
  • Duplicate Detection: Find similar or duplicate content
""")