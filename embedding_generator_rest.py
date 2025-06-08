import os
import json
from typing import List, Dict
import openai
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Supabase REST API
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")
        
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
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
    
    def store_chunk_in_supabase(self, chunk_data: Dict) -> bool:
        """Store a single chunk with its embedding in Supabase via REST API"""
        try:
            # Generate embedding for the chunk text
            embedding = self.generate_embedding(chunk_data["chunk_text"])
            
            if embedding is None:
                print(f"Failed to generate embedding for chunk: {chunk_data['section_title']}")
                return False
            
            # Prepare data for insertion
            insert_data = {
                "manual_id": chunk_data["manual_id"],
                "section_title": chunk_data["section_title"],
                "chunk_text": chunk_data["chunk_text"],
                "embedding_vector": embedding
            }
            
            # Insert via REST API
            url = f"{self.supabase_url}/rest/v1/documents"
            response = requests.post(url, headers=self.headers, json=insert_data)
            
            if response.status_code in [200, 201]:
                print(f"✓ Stored chunk: {chunk_data['section_title'][:50]}...")
                return True
            else:
                print(f"✗ Failed to store chunk: {chunk_data['section_title']} - Status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error storing chunk: {e}")
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
                time.sleep(0.5)
            
            # Wait between batches
            if i + batch_size < total_chunks:
                print("Waiting between batches...")
                time.sleep(2)
        
        return {
            "total": total_chunks,
            "successful": successful,
            "failed": failed
        }
    
    def test_supabase_connection(self) -> bool:
        """Test the Supabase REST API connection"""
        try:
            url = f"{self.supabase_url}/rest/v1/documents?select=id&limit=1"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                print("✓ Supabase REST API connection successful")
                return True
            else:
                print(f"✗ Supabase connection failed - Status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Supabase connection failed: {e}")
            return False
    
    def clear_existing_data(self, manual_ids: List[str] = None) -> bool:
        """Clear existing data for specific manuals (useful for re-processing)"""
        try:
            if manual_ids:
                for manual_id in manual_ids:
                    url = f"{self.supabase_url}/rest/v1/documents?manual_id=eq.{manual_id}"
                    response = requests.delete(url, headers=self.headers)
                    if response.status_code in [200, 204]:
                        print(f"Cleared existing data for manual: {manual_id}")
                    else:
                        print(f"Failed to clear data for manual: {manual_id}")
            else:
                # Clear all data (be careful with this!)
                url = f"{self.supabase_url}/rest/v1/documents"
                response = requests.delete(url, headers=self.headers)
                if response.status_code in [200, 204]:
                    print("Cleared all existing data")
                else:
                    print("Failed to clear all data")
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
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
        url = f"{generator.supabase_url}/rest/v1/documents?select=manual_id,section_title&limit=5"
        response = requests.get(url, headers=generator.headers)
        
        if response.status_code == 200:
            docs = response.json()
            print(f"\nSample stored documents:")
            for doc in docs:
                print(f"  - {doc['manual_id']}: {doc['section_title']}")
        else:
            print(f"Error fetching sample documents: {response.status_code}")
    except Exception as e:
        print(f"Error fetching sample documents: {e}")

if __name__ == "__main__":
    main()