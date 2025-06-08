import os
import json
from typing import List, Dict
import openai
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize PostgreSQL connection (Supabase uses PostgreSQL)
        supabase_url = os.getenv("SUPABASE_URL")
        
        if not supabase_url:
            raise ValueError("Please set SUPABASE_URL environment variable")
        
        # Parse Supabase URL to get connection details
        # Format: https://projectref.supabase.co
        project_ref = supabase_url.replace("https://", "").replace(".supabase.co", "")
        
        # Supabase PostgreSQL connection details
        self.conn_params = {
            "host": f"db.{project_ref}.supabase.co",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "d4uNjsJIa3l2RW5g"  # You'll need to provide your actual DB password
        }
        
        self.conn = None
        
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
    
    def connect_to_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def store_chunk_in_supabase(self, chunk_data: Dict) -> bool:
        """Store a single chunk with its embedding in Supabase"""
        try:
            # Generate embedding for the chunk text
            embedding = self.generate_embedding(chunk_data["chunk_text"])
            
            if embedding is None:
                print(f"Failed to generate embedding for chunk: {chunk_data['section_title']}")
                return False
            
            # Ensure database connection
            if not self.conn or self.conn.closed:
                if not self.connect_to_db():
                    return False
            
            # Insert into PostgreSQL
            cursor = self.conn.cursor()
            
            insert_query = """
            INSERT INTO documents (manual_id, section_title, chunk_text, embedding_vector)
            VALUES (%s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                chunk_data["manual_id"],
                chunk_data["section_title"],
                chunk_data["chunk_text"],
                embedding
            ))
            
            self.conn.commit()
            cursor.close()
            
            print(f"✓ Stored chunk: {chunk_data['section_title'][:50]}...")
            return True
                
        except Exception as e:
            print(f"Error storing chunk in database: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def batch_process_chunks(self, chunks: List[Dict], batch_size: int = 5) -> Dict:
        """Process chunks in batches to avoid rate limits"""
        total_chunks = len(chunks)
        successful = 0
        failed = 0
        
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            print(f"\nProcessing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}")
            
            for chunk in batch:
                if self.store_chunk_in_supabase(chunk):
                    successful += 1
                else:
                    failed += 1
                
                # Rate limiting - wait between requests
                time.sleep(0.2)
            
            # Wait between batches
            if i + batch_size < total_chunks:
                print("Waiting between batches...")
                time.sleep(1)
        
        return {
            "total": total_chunks,
            "successful": successful,
            "failed": failed
        }
    
    def test_supabase_connection(self) -> bool:
        """Test the database connection"""
        try:
            if self.connect_to_db():
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                print("✓ Database connection successful")
                return True
            else:
                return False
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def clear_existing_data(self, manual_ids: List[str] = None) -> bool:
        """Clear existing data for specific manuals (useful for re-processing)"""
        try:
            if not self.conn or self.conn.closed:
                if not self.connect_to_db():
                    return False
            
            cursor = self.conn.cursor()
            
            if manual_ids:
                for manual_id in manual_ids:
                    cursor.execute("DELETE FROM documents WHERE manual_id = %s", (manual_id,))
                    print(f"Cleared existing data for manual: {manual_id}")
            else:
                cursor.execute("DELETE FROM documents")
                print("Cleared all existing data")
            
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            if self.conn:
                self.conn.rollback()
            return False

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
    
    # Test Supabase connection
    if not generator.test_supabase_connection():
        print("Failed to connect to Supabase. Please check your credentials.")
        return
    
    # Ask user if they want to clear existing data
    manual_ids = list(set(chunk["manual_id"] for chunk in chunks))
    print(f"\nFound manuals: {manual_ids}")
    
    clear_data = input("Do you want to clear existing data for these manuals? (y/n): ").lower().strip()
    if clear_data == 'y':
        generator.clear_existing_data(manual_ids)
    
    # Process chunks and generate embeddings
    print(f"\nStarting embedding generation for {len(chunks)} chunks...")
    
    results = generator.batch_process_chunks(chunks)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total chunks: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['successful']/results['total']*100:.1f}%")
    
    # Test query to verify data was stored
    try:
        if generator.conn and not generator.conn.closed:
            cursor = generator.conn.cursor()
            cursor.execute("SELECT manual_id, section_title FROM documents LIMIT 5")
            rows = cursor.fetchall()
            
            print(f"\nSample stored documents:")
            for row in rows:
                print(f"  - {row[0]}: {row[1]}")
            
            cursor.close()
    except Exception as e:
        print(f"Error fetching sample documents: {e}")
    
    # Close database connection
    if generator.conn:
        generator.conn.close()

if __name__ == "__main__":
    main()