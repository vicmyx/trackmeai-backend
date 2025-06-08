import os
import json
from typing import List, Dict
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI text-embedding-3-small"""
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
    
    def process_chunk(self, chunk_data: Dict) -> Dict:
        """Process a single chunk and add embedding"""
        try:
            print(f"Processing: {chunk_data['section_title'][:50]}...")
            
            # Generate embedding for the chunk text
            embedding = self.generate_embedding(chunk_data["chunk_text"])
            
            if embedding is None:
                print(f"Failed to generate embedding for chunk: {chunk_data['section_title']}")
                return None
            
            # Add embedding to chunk data
            result = chunk_data.copy()
            result["embedding_vector"] = embedding
            result["embedding_dimensions"] = len(embedding)
            
            print(f"✓ Generated embedding ({len(embedding)} dimensions)")
            return result
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None
    
    def batch_process_chunks(self, chunks: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Process chunks in batches to avoid rate limits"""
        total_chunks = len(chunks)
        processed_chunks = []
        
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            print(f"\nProcessing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}")
            
            for chunk in batch:
                processed_chunk = self.process_chunk(chunk)
                if processed_chunk:
                    processed_chunks.append(processed_chunk)
                
                # Rate limiting - wait between requests to avoid OpenAI rate limits
                time.sleep(0.5)
            
            # Wait between batches
            if i + batch_size < total_chunks:
                print("Waiting between batches...")
                time.sleep(2)
        
        return processed_chunks

def main():
    # Load processed chunks
    try:
        with open("processed_chunks.json", "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: processed_chunks.json not found. Please run pdf_processor.py first.")
        return
    
    print(f"Loaded {len(chunks)} chunks from processed_chunks.json")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Process chunks and generate embeddings
    print(f"\nStarting embedding generation for {len(chunks)} chunks...")
    
    processed_chunks = generator.batch_process_chunks(chunks)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total input chunks: {len(chunks)}")
    print(f"Successfully processed: {len(processed_chunks)}")
    print(f"Failed: {len(chunks) - len(processed_chunks)}")
    
    if processed_chunks:
        print(f"Success rate: {len(processed_chunks)/len(chunks)*100:.1f}%")
        
        # Save processed chunks with embeddings to file
        output_file = "chunks_with_embeddings.json"
        with open(output_file, "w") as f:
            json.dump(processed_chunks, f, indent=2)
        
        print(f"\nEmbeddings saved to: {output_file}")
        
        # Show some stats
        first_chunk = processed_chunks[0]
        print(f"\nEmbedding info:")
        print(f"  - Dimensions: {first_chunk['embedding_dimensions']}")
        print(f"  - Model: text-embedding-3-small")
        print(f"  - Total vectors generated: {len(processed_chunks)}")
        
        # Create SQL file for manual import
        create_sql_file(processed_chunks)
    else:
        print("No chunks were successfully processed.")

def create_sql_file(chunks: List[Dict]):
    """Create SQL file for manual database import"""
    sql_lines = []
    
    # Add table creation SQL
    sql_lines.append("-- Create documents table with pgvector extension")
    sql_lines.append("CREATE EXTENSION IF NOT EXISTS vector;")
    sql_lines.append("")
    sql_lines.append("""CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    manual_id TEXT NOT NULL,
    section_title TEXT,
    chunk_text TEXT NOT NULL,
    embedding_vector VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);""")
    sql_lines.append("")
    sql_lines.append("CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);")
    sql_lines.append("")
    sql_lines.append("-- Insert document chunks")
    
    for chunk in chunks:
        # Escape single quotes in text
        manual_id = chunk["manual_id"].replace("'", "''")
        section_title = chunk["section_title"].replace("'", "''")
        chunk_text = chunk["chunk_text"].replace("'", "''")
        embedding_str = str(chunk["embedding_vector"])
        
        sql = f"INSERT INTO documents (manual_id, section_title, chunk_text, embedding_vector) VALUES ('{manual_id}', '{section_title}', '{chunk_text}', '{embedding_str}');"
        sql_lines.append(sql)
    
    # Save to file
    with open("documents_import.sql", "w") as f:
        f.write("\n".join(sql_lines))
    
    print("SQL import file created: documents_import.sql")
    print("You can run this in your Supabase SQL editor to import the data.")

if __name__ == "__main__":
    main()