#!/usr/bin/env python3

from ai_mechanic_core import AIMechanicCore

def debug_prerace_response():
    """Debug why the AI is not providing a proper pre-race checklist response"""
    
    print("=== DEBUGGING PRE-RACE CHECKLIST RESPONSE ===\n")
    
    ai_mechanic = AIMechanicCore(embeddings_file="chunks_with_embeddings_final.json")
    
    # Get the similar chunks
    question = "What is the pre-race checklist?"
    similar_chunks = ai_mechanic.vector_similarity_search(question, top_k=7)
    
    # Build the context prompt
    context_prompt = ai_mechanic.build_context_prompt(question, similar_chunks)
    
    print("=== CONTEXT PROMPT BEING SENT TO GPT-4O ===")
    print(context_prompt)
    print("\n" + "="*80 + "\n")
    
    # Get the response
    response = ai_mechanic.query_gpt4o(context_prompt)
    print("=== GPT-4O RESPONSE ===")
    print(response)
    print("\n" + "="*80 + "\n")
    
    # Test with a more specific question
    print("=== TESTING WITH MORE SPECIFIC QUESTION ===")
    specific_question = "What should I check in the engine bay before racing?"
    
    specific_chunks = ai_mechanic.vector_similarity_search(specific_question, top_k=5)
    specific_prompt = ai_mechanic.build_context_prompt(specific_question, specific_chunks)
    
    print("Specific question context:")
    print(specific_prompt[:500] + "...")
    print("\nSpecific response:")
    specific_response = ai_mechanic.query_gpt4o(specific_prompt)
    print(specific_response)

if __name__ == "__main__":
    debug_prerace_response()