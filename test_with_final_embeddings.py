#!/usr/bin/env python3
"""
Test AI mechanic with the final embeddings file that has more documents
"""

from ai_mechanic_core_enhanced import EnhancedAIMechanicCore

def test_with_final_embeddings():
    """Test AI mechanic with final embeddings file"""
    
    print("🔧 TESTING WITH FINAL EMBEDDINGS FILE")
    print("=" * 50)
    
    # Initialize AI mechanic with final embeddings file
    try:
        ai_mechanic = EnhancedAIMechanicCore("chunks_with_embeddings_final.json")
        print("✅ AI Mechanic initialized with final embeddings")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test the same questions that failed before
    failed_questions = [
        "What tires should I use?",
        "What are the engine specifications?", 
        "How do I adjust the suspension?",
        "What oil should I use?",
        "What is the gear ratio?"
    ]
    
    print(f"\n🧪 Testing {len(failed_questions)} previously failed questions:")
    
    for i, question in enumerate(failed_questions, 1):
        print(f"\n{i}. TESTING: '{question}'")
        print("-" * 40)
        
        try:
            response = ai_mechanic.ask_ai_mechanic(question)
            
            answer = response['answer']
            source_count = len(response['source_chunks'])
            confidence = response['total_confidence']
            
            # Check if it's a "manual does not specify" response
            is_not_specified = "manual does not specify" in answer.lower()
            
            print(f"📤 Answer preview: {answer[:150]}...")
            print(f"📊 Sources: {source_count}, Confidence: {confidence:.3f}")
            print(f"🎯 Valid answer: {'❌' if is_not_specified else '✅'}")
            
            if not is_not_specified:
                print(f"✅ SUCCESS: Now returning valid answer!")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_with_final_embeddings()