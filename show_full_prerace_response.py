#!/usr/bin/env python3

from ai_mechanic_core_improved import AIMechanicCoreImproved

def show_full_response():
    """Show the complete pre-race checklist response"""
    
    print("=== COMPLETE PRE-RACE CHECKLIST RESPONSE ===\n")
    
    ai_mechanic = AIMechanicCoreImproved(embeddings_file="chunks_with_embeddings_final.json")
    response = ai_mechanic.ask_ai_mechanic("What is the pre-race checklist?", top_k=7)
    
    print("QUESTION:")
    print(response['question'])
    print("\n" + "="*80 + "\n")
    
    print("COMPLETE ANSWER:")
    print(response['answer'])
    print("\n" + "="*80 + "\n")
    
    print("SIMILARITY SCORES AND SOURCE VERIFICATION:")
    for i, chunk in enumerate(response['source_chunks'], 1):
        print(f"{i}. {chunk['section_title']}")
        print(f"   Manual: {chunk['manual_id']}")
        print(f"   Similarity Score: {chunk['similarity_score']:.4f}")
        print(f"   Content Length: {len(chunk['chunk_text'])} characters")
        print()

if __name__ == "__main__":
    show_full_response()