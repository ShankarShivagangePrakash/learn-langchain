# =============================================================================
# TEXT SIMILARITY FINDER USING VECTOR EMBEDDINGS
# =============================================================================
"""
Educational Demo: Text Similarity Analysis Using OpenAI Embeddings

This script demonstrates how to calculate semantic similarity between two text 
inputs using vector embeddings. It showcases the practical application of 
embeddings in natural language processing for measuring text relatedness.

TECHNICAL CONCEPTS COVERED:
• OpenAI text-embedding-ada-002 model usage
• Vector embeddings generation (1536 dimensions)
• Dot product similarity calculation
• Semantic similarity vs. lexical similarity
• High-dimensional vector mathematics

EDUCATIONAL VALUE:
• Understand how AI models represent text as numerical vectors
• Learn practical similarity measurement techniques
• Explore real-world applications of embedding technology
• Compare semantic understanding vs. keyword matching

BUSINESS APPLICATIONS:
• Document similarity analysis
• Content recommendation systems
• Duplicate detection algorithms
• Search relevance scoring
• Customer query matching
"""

import os
from langchain_openai import OpenAIEmbeddings
import numpy as np

# =============================================================================
# ENVIRONMENT CONFIGURATION & MODEL INITIALIZATION
# =============================================================================

# Retrieve OpenAI API key from environment variables
# Security Best Practice: Never hardcode API keys in source code
# Set environment variable: export OPENAI_API_KEY="your-api-key-here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings model
# Technical Details:
# - Model: text-embedding-ada-002 (OpenAI's most capable embedding model)
# - Output Dimensions: 1536 (high-dimensional vector space)
# - Token Limit: ~8,191 tokens per input
# - Cost: $0.0001 per 1K tokens (as of 2024)
# - Use Cases: Semantic search, similarity, clustering, classification
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# =============================================================================
# USER INPUT COLLECTION & VALIDATION
# =============================================================================

print("🔍 Text Similarity Finder Using Vector Embeddings")
print("=" * 50)
print("Compare the semantic similarity between two pieces of text.")
print("Examples to try:")
print("• 'I love programming' vs 'I enjoy coding'")
print("• 'The weather is sunny' vs 'It's a bright day'")
print("• 'Machine learning' vs 'Artificial intelligence'")
print("• 'Dog' vs 'Cat' (for contrast)")
print()

# Collect first text input from user
# Educational Note: The model can handle various text lengths and formats
# - Single words, phrases, sentences, or paragraphs
# - Technical terms, colloquial language, or formal text
# - Multi-language support (though performance varies by language)
text1 = input("📝 Enter the first text: ").strip()

# Collect second text input for comparison
text2 = input("📝 Enter the second text: ").strip()

# Input validation
if not text1 or not text2:
    print("❌ Error: Both text inputs are required.")
    print("💡 Please provide two non-empty text strings for comparison.")
    exit(1)

print(f"\n🔄 Analyzing similarity between:")
print(f"   Text 1: '{text1}'")
print(f"   Text 2: '{text2}'")

# =============================================================================
# VECTOR EMBEDDING GENERATION
# =============================================================================

print(f"\n⚡ Generating embeddings...")
print(f"📊 Converting text to 1536-dimensional vectors...")

# Generate embedding vector for first text
# Technical Process:
# 1. Text tokenization (breaking into subword tokens)
# 2. Token encoding (converting to numerical IDs)
# 3. Model processing through transformer architecture
# 4. Output: Dense vector representation capturing semantic meaning
# 5. Vector normalization for consistent similarity calculations
response1 = embeddings_model.embed_query(text1)

# Generate embedding vector for second text
# Note: Each embed_query() call makes an API request to OpenAI
# Cost Consideration: Each call consumes tokens based on input length
# Performance: Typical response time 100-500ms depending on text length
response2 = embeddings_model.embed_query(text2)

# =============================================================================
# SIMILARITY CALCULATION & ANALYSIS
# =============================================================================

print(f"✅ Embeddings generated successfully!")
print(f"📐 Calculating semantic similarity...")

# Calculate dot product similarity score
# Mathematical Explanation:
# - Dot product: sum of element-wise multiplication of two vectors
# - Formula: similarity = Σ(a_i × b_i) for i = 1 to n
# - Higher values indicate greater similarity
# - Range: Typically -1 to 1 for normalized vectors
# - OpenAI embeddings are normalized, so range is roughly 0.0 to 1.0
similarity_score = np.dot(response1, response2)

# =============================================================================
# RESULTS DISPLAY & INTERPRETATION
# =============================================================================

# Convert similarity score to percentage for user-friendly display
# Technical Note: Multiplying by 100 for percentage representation
# The original similarity score (0.0-1.0) becomes (0%-100%)
similarity_percentage = similarity_score * 100

print(f"\n🎯 **SIMILARITY ANALYSIS RESULTS**")
print(f"=" * 40)
print(f"📊 Similarity Score: {similarity_score:.4f}")
print(f"📈 Similarity Percentage: {similarity_percentage:.2f}%")

# Provide interpretation of the similarity score
# Educational Guidance: Help users understand what the numbers mean
if similarity_percentage >= 90:
    interpretation = "🟢 EXTREMELY SIMILAR - Nearly identical meaning"
    explanation = "These texts express very similar or identical concepts."
elif similarity_percentage >= 75:
    interpretation = "🔵 HIGHLY SIMILAR - Strong semantic relationship"
    explanation = "These texts are closely related in meaning and context."
elif similarity_percentage >= 60:
    interpretation = "🟡 MODERATELY SIMILAR - Some semantic overlap"
    explanation = "These texts share some common themes or concepts."
elif similarity_percentage >= 40:
    interpretation = "🟠 SOMEWHAT SIMILAR - Limited semantic connection"
    explanation = "These texts have minor similarities but different overall meanings."
else:
    interpretation = "🔴 DISSIMILAR - Little to no semantic relationship"
    explanation = "These texts express different concepts with minimal overlap."

print(f"\n{interpretation}")
print(f"💡 {explanation}")

# =============================================================================
# EDUCATIONAL INSIGHTS & TECHNICAL DETAILS
# =============================================================================

print(f"\n📚 **UNDERSTANDING THE RESULTS:**")
print(f"")
print(f"🔬 **How Similarity Was Calculated:**")
print(f"   1. Each text converted to 1536-dimensional vector")
print(f"   2. Vectors capture semantic meaning, not just keywords")
print(f"   3. Dot product calculated between the two vectors")
print(f"   4. Higher scores = more similar meanings")
print(f"")
print(f"🧠 **Why This Works:**")
print(f"   • Embeddings capture context and meaning")
print(f"   • Similar concepts cluster together in vector space")
print(f"   • Mathematical operations reveal semantic relationships")
print(f"   • Model trained on vast text data learns word associations")

print(f"\n🔍 **COMPARISON WITH OTHER METHODS:**")
print(f"")
print(f"📝 Keyword Matching:")
print(f"   • Only finds exact word matches")
print(f"   • Misses synonyms and related concepts")
print(f"   • Example: 'car' vs 'automobile' = 0% similarity")
print(f"")
print(f"🤖 Semantic Embeddings (This Method):")
print(f"   • Understands meaning and context")
print(f"   • Recognizes synonyms and related terms")
print(f"   • Example: 'car' vs 'automobile' = ~85% similarity")

# Display technical details about the embedding vectors
print(f"\n⚙️ **TECHNICAL DETAILS:**")
print(f"   • Embedding Model: text-embedding-ada-002")
print(f"   • Vector Dimensions: {len(response1)}")
print(f"   • Vector Length (Text 1): {np.linalg.norm(response1):.4f}")
print(f"   • Vector Length (Text 2): {np.linalg.norm(response2):.4f}")
print(f"   • Similarity Metric: Dot Product")
print(f"   • Score Range: 0.0 (dissimilar) to 1.0 (identical)")

# =============================================================================
# PRACTICAL APPLICATIONS & USE CASES
# =============================================================================

print(f"\n🚀 **REAL-WORLD APPLICATIONS:**")
print(f"""
🏢 BUSINESS USE CASES:
• Document similarity analysis for knowledge management
• Customer support query routing and matching
• Content recommendation systems (articles, products)
• Duplicate content detection and deduplication
• Search result relevance scoring and ranking

🔬 RESEARCH & ACADEMIC:
• Literature review and paper similarity analysis
• Plagiarism detection and academic integrity
• Research topic clustering and categorization
• Citation analysis and reference matching

💻 DEVELOPMENT & TECHNICAL:
• Code similarity detection and refactoring
• API documentation matching and discovery
• Bug report similarity for issue tracking
• Feature request categorization and prioritization

📊 DATA SCIENCE & ANALYTICS:
• Customer feedback sentiment and topic analysis
• Social media content categorization
• Survey response similarity and clustering
• Market research and competitive analysis
""")

print(f"\n✨ **TRY MORE COMPARISONS:**")
print(f"   • Technical terms vs. plain language explanations")
print(f"   • Different languages expressing same concepts")
print(f"   • Formal vs. informal expressions of ideas")
print(f"   • Product descriptions vs. customer reviews")
print(f"   • Questions vs. potential answers")

print(f"\n🎯 **TIPS FOR BETTER RESULTS:**")
print(f"   • Use complete sentences for better context")
print(f"   • Include relevant keywords and concepts")
print(f"   • Consider the domain and context of your text")
print(f"   • Longer texts generally provide more context")
print(f"   • Be specific about the concepts you're comparing")