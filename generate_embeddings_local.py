import os
import json
from typing import List, Dict
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class LocalEmbeddingGenerator:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_chunks_with_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to chunks"""
        chunks_with_embeddings = []
        total = len(chunks)
        
        print(f"Generating embeddings for {total} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total}: {chunk['section_title'][:50]}...")
            
            # Generate embedding
            embedding = self.generate_embedding(chunk["chunk_text"])
            
            if embedding is None:
                print(f"Failed to generate embedding for chunk {i}")
                continue
            
            # Add embedding to chunk
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding_vector"] = embedding
            chunk_with_embedding["embedding_dimensions"] = len(embedding)
            
            chunks_with_embeddings.append(chunk_with_embedding)
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings

def main():
    # Load the final processed chunks
    try:
        with open("processed_chunks_final.json", "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: processed_chunks_final.json not found. Please run pdf_processor_final.py first.")
        return
    
    print(f"Loaded {len(chunks)} chunks from processed_chunks_final.json")
    
    # Initialize embedding generator
    generator = LocalEmbeddingGenerator()
    
    # Generate embeddings
    chunks_with_embeddings = generator.process_chunks_with_embeddings(chunks)
    
    if not chunks_with_embeddings:
        print("No embeddings were generated successfully.")
        return
    
    # Save chunks with embeddings
    output_file = "chunks_with_embeddings_final.json"
    with open(output_file, "w") as f:
        json.dump(chunks_with_embeddings, f, indent=2)
    
    print(f"\nChunks with embeddings saved to {output_file}")
    
    # Show statistics
    print(f"\n=== STATISTICS ===")
    print(f"Total chunks with embeddings: {len(chunks_with_embeddings)}")
    
    word_counts = [c['word_count'] for c in chunks_with_embeddings]
    if word_counts:
        print(f"Average words per chunk: {sum(word_counts)/len(word_counts):.1f}")
        print(f"Chunks with >50 words: {len([c for c in chunks_with_embeddings if c['word_count'] > 50])}")
    
    # Test a simple search
    print(f"\n=== TESTING SEARCH ===")
    test_query = "brake bleeding procedure"
    print(f"Test query: '{test_query}'")
    
    # Generate query embedding
    query_embedding = generator.generate_embedding(test_query)
    if query_embedding:
        # Simple similarity search
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get all embeddings
        embeddings_matrix = np.array([c["embedding_vector"] for c in chunks_with_embeddings])
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        
        # Get top 3 results
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print("Top 3 results:")
        for i, idx in enumerate(top_indices, 1):
            chunk = chunks_with_embeddings[idx]
            print(f"{i}. {chunk['section_title']} (similarity: {similarities[idx]:.3f})")
            print(f"   Content: {chunk['chunk_text'][:100]}...")
            print()

if __name__ == "__main__":
    main()