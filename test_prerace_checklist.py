#!/usr/bin/env python3

import json
from ai_mechanic_core import AIMechanicCore

def test_prerace_checklist():
    """Test the AI mechanic system with the pre-race checklist question"""
    
    print("=== Testing AI Mechanic with Pre-Race Checklist Question ===\n")
    
    try:
        # Initialize AI Mechanic with the final embeddings file
        print("Initializing AI Mechanic with chunks_with_embeddings_final.json...")
        ai_mechanic = AIMechanicCore(embeddings_file="chunks_with_embeddings_final.json")
        
        # Test question
        question = "What is the pre-race checklist?"
        
        print(f"Testing question: '{question}'\n")
        
        # Get the response
        response = ai_mechanic.ask_ai_mechanic(question, top_k=7)
        
        # Display detailed results
        print("=== DETAILED TEST RESULTS ===\n")
        
        print(f"Question: {response['question']}\n")
        
        print("=== SIMILARITY SCORES AND RETRIEVED CONTENT ===")
        for i, chunk in enumerate(response['source_chunks'], 1):
            print(f"\n--- Source {i} ---")
            print(f"Manual: {chunk['manual_id']}")
            print(f"Section: {chunk['section_title']}")
            print(f"Similarity Score: {chunk['similarity_score']:.4f}")
            print(f"Content Preview: {chunk['chunk_text'][:200]}...")
            if len(chunk['chunk_text']) > 200:
                print(f"[Full content length: {len(chunk['chunk_text'])} characters]")
        
        print(f"\n=== FINAL AI RESPONSE ===")
        print(f"Answer: {response['answer']}\n")
        
        # Analyze response quality
        print("=== RESPONSE QUALITY ANALYSIS ===")
        answer_lower = response['answer'].lower()
        
        if "the manual does not specify" in answer_lower:
            print("⚠️  WARNING: AI returned generic 'manual does not specify' response")
            print("This suggests the retrieval or context building may need improvement")
        else:
            print("✅ AI provided a specific answer based on manual content")
        
        # Check if any high similarity scores were found
        high_sim_count = sum(1 for chunk in response['source_chunks'] if chunk['similarity_score'] > 0.5)
        med_sim_count = sum(1 for chunk in response['source_chunks'] if 0.3 < chunk['similarity_score'] <= 0.5)
        low_sim_count = sum(1 for chunk in response['source_chunks'] if chunk['similarity_score'] <= 0.3)
        
        print(f"\nSimilarity Score Distribution:")
        print(f"  High similarity (>0.5): {high_sim_count} chunks")
        print(f"  Medium similarity (0.3-0.5): {med_sim_count} chunks")
        print(f"  Low similarity (≤0.3): {low_sim_count} chunks")
        
        # Check for specific pre-race related terms in retrieved content
        prerace_terms = ['pre-race', 'checklist', 'before', 'start', 'inspection', 'safety', 'preparation']
        
        print(f"\nRelevance Analysis:")
        for term in prerace_terms:
            found_in = []
            for i, chunk in enumerate(response['source_chunks'], 1):
                if term.lower() in chunk['chunk_text'].lower() or term.lower() in chunk['section_title'].lower():
                    found_in.append(f"Source {i}")
            
            if found_in:
                print(f"  '{term}': Found in {', '.join(found_in)}")
        
        return response
        
    except Exception as e:
        print(f"Error during test: {e}")
        return None

def examine_embeddings_file():
    """Examine the structure of the final embeddings file"""
    try:
        print("\n=== EXAMINING EMBEDDINGS FILE STRUCTURE ===")
        
        with open("chunks_with_embeddings_final.json", "r") as f:
            data = json.load(f)
        
        print(f"Total documents: {len(data)}")
        
        # Analyze by manual
        manual_counts = {}
        for doc in data:
            manual_id = doc.get('manual_id', 'Unknown')
            manual_counts[manual_id] = manual_counts.get(manual_id, 0) + 1
        
        print("Documents by manual:")
        for manual, count in manual_counts.items():
            print(f"  {manual}: {count} chunks")
        
        # Look for pre-race related sections
        print("\nSections that might contain pre-race information:")
        prerace_keywords = ['pre-race', 'checklist', 'before', 'start', 'preparation', 'inspection', 'safety']
        
        relevant_sections = []
        for doc in data:
            section_title = doc.get('section_title', '').lower()
            chunk_text = doc.get('chunk_text', '').lower()
            
            for keyword in prerace_keywords:
                if keyword in section_title or keyword in chunk_text:
                    relevant_sections.append({
                        'manual': doc.get('manual_id'),
                        'section': doc.get('section_title'),
                        'keyword': keyword
                    })
                    break
        
        print(f"Found {len(relevant_sections)} potentially relevant sections:")
        for section in relevant_sections[:10]:  # Show first 10
            print(f"  {section['manual']} - {section['section']} (matched: {section['keyword']})")
        
        if len(relevant_sections) > 10:
            print(f"  ... and {len(relevant_sections) - 10} more")
            
    except Exception as e:
        print(f"Error examining embeddings file: {e}")

if __name__ == "__main__":
    # First examine the embeddings file
    examine_embeddings_file()
    
    # Then test the pre-race question
    test_prerace_checklist()