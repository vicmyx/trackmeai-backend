import os
import json
from typing import List, Dict, Tuple
import openai
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class AIMechanicCoreImproved:
    def __init__(self, embeddings_file: str = "chunks_with_embeddings_final.json"):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Load embeddings data
        self.load_embeddings(embeddings_file)
        
    def load_embeddings(self, embeddings_file: str):
        """Load the embeddings data from file"""
        try:
            with open(embeddings_file, "r") as f:
                self.documents = json.load(f)
            
            # Convert embeddings to numpy arrays for faster similarity search
            self.embeddings_matrix = np.array([doc["embedding_vector"] for doc in self.documents])
            
            print(f"Loaded {len(self.documents)} documents with embeddings")
            print(f"Embedding dimensions: {self.embeddings_matrix.shape[1]}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file {embeddings_file} not found. Please run embedding generation first.")
        except Exception as e:
            raise Exception(f"Error loading embeddings: {e}")
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a user query"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            raise Exception(f"Error generating query embedding: {e}")
    
    def vector_similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search to find most relevant chunks"""
        # Generate embedding for the query
        query_embedding = self.generate_query_embedding(query)
        
        # Calculate cosine similarity between query and all document embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = float(similarities[idx])
            results.append(doc)
        
        return results
    
    def build_context_prompt(self, query: str, similar_chunks: List[Dict]) -> str:
        """Build the context prompt for GPT-4o with improved instructions"""
        context_sections = []
        
        for i, chunk in enumerate(similar_chunks, 1):
            context_sections.append(f"**Source {i} - {chunk['manual_id']} - {chunk['section_title']}:**\n{chunk['chunk_text']}")
        
        context = "\n\n".join(context_sections)
        
        # Detect if this is a checklist-type question
        is_checklist_question = any(term in query.lower() for term in ['checklist', 'check', 'inspect', 'prepare', 'setup'])
        
        if is_checklist_question:
            prompt = f"""You are an AI Mechanic trained on the Radical SR3 Owner's Manual and Handling Guide.
Answer the following question using the provided context from the manuals.

**Context from Manuals:**
{context}

**Question:** {query}

**Instructions:**
- Answer based on the information provided in the context above
- When asked about checklists or procedures, synthesize information from multiple sections to provide a comprehensive answer
- The manual contains maintenance and inspection procedures across different sections - these collectively form practical checklists
- Be specific and technical when the manual provides detailed information
- Reference the specific manual sections when relevant
- If the context contains relevant procedural information but not an explicit checklist title, organize the information into a logical checklist format
- Only say "The manual does not specify" if the context truly contains no relevant information for the question

**Answer:**"""
        else:
            prompt = f"""You are an AI Mechanic trained on the Radical SR3 Owner's Manual and Handling Guide.
Answer the following question only using the provided context from the manuals.

**Context from Manuals:**
{context}

**Question:** {query}

**Instructions:**
- Answer based ONLY on the information provided in the context above
- If the manual does not provide specific information for this question, say: "The manual does not specify this information. Please consult your team or manufacturer."
- Be specific and technical when the manual provides detailed information
- Reference the specific manual sections when relevant
- If multiple manuals provide different information, mention both

**Answer:**"""
        
        return prompt
    
    def query_gpt4o(self, prompt: str) -> str:
        """Query GPT-4o with the context prompt"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert AI Mechanic specializing in Radical SR3 race cars. Provide accurate, technical answers based on the provided manual content. When answering checklist questions, synthesize information from multiple sections to provide comprehensive, actionable guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2  # Slightly higher temperature for better synthesis
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error querying GPT-4o: {e}")
    
    def ask_ai_mechanic(self, question: str, top_k: int = 7) -> Dict:
        """Main function to ask the AI Mechanic a question"""
        try:
            print(f"Question: {question}")
            print(f"Searching for relevant information...")
            
            # Step 1: Vector similarity search
            similar_chunks = self.vector_similarity_search(question, top_k)
            
            print(f"Found {len(similar_chunks)} relevant sections:")
            for i, chunk in enumerate(similar_chunks, 1):
                print(f"  {i}. {chunk['section_title']} (score: {chunk['similarity_score']:.3f})")
            
            # Step 2: Build context prompt
            context_prompt = self.build_context_prompt(question, similar_chunks)
            
            # Step 3: Query GPT-4o
            print("Generating answer...")
            answer = self.query_gpt4o(context_prompt)
            
            # Step 4: Prepare response
            response = {
                "question": question,
                "answer": answer,
                "source_chunks": [
                    {
                        "manual_id": chunk["manual_id"],
                        "section_title": chunk["section_title"],
                        "chunk_text": chunk["chunk_text"],
                        "similarity_score": chunk["similarity_score"]
                    }
                    for chunk in similar_chunks
                ],
                "top_k": top_k
            }
            
            return response
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "source_chunks": [],
                "top_k": top_k,
                "error": True
            }

def main():
    """Test the improved AI Mechanic system"""
    try:
        ai_mechanic = AIMechanicCoreImproved()
        
        # Test the pre-race checklist question
        question = "What is the pre-race checklist?"
        
        print("=== Testing Improved AI Mechanic ===\n")
        
        response = ai_mechanic.ask_ai_mechanic(question)
        
        print(f"Q: {response['question']}")
        print(f"A: {response['answer']}")
        print(f"\nSources used:")
        for j, source in enumerate(response['source_chunks'], 1):
            print(f"  {j}. {source['section_title']} (similarity: {source['similarity_score']:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()