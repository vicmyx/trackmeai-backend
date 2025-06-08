#!/usr/bin/env python3
"""
Test multiple questions to identify patterns in AI mechanic responses
"""

from ai_mechanic_core_enhanced import EnhancedAIMechanicCore

def test_questions():
    """Test various questions to identify issue patterns"""
    
    print("🔧 TESTING MULTIPLE QUESTIONS")
    print("=" * 50)
    
    # Initialize AI mechanic
    try:
        ai_mechanic = EnhancedAIMechanicCore("chunks_with_embeddings_advanced.json")
        print("✅ AI Mechanic initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test questions
    test_questions = [
        "What is the pre-race checklist?",
        "How do I check the brakes?",
        "What tires should I use?",
        "How do I clean the car?",
        "What are the engine specifications?",
        "How do I adjust the suspension?",
        "What oil should I use?",
        "How do I perform maintenance?",
        "What is the gear ratio?",
        "How do I check the differential?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. TESTING: '{question}'")
        print("-" * 40)
        
        try:
            response = ai_mechanic.ask_ai_mechanic(question)
            
            answer = response['answer']
            source_count = len(response['source_chunks'])
            confidence = response['total_confidence']
            has_sources = response['has_trackmeai_sources']
            
            # Check if it's a "manual does not specify" response
            is_not_specified = "manual does not specify" in answer.lower()
            
            results.append({
                'question': question,
                'is_not_specified': is_not_specified,
                'source_count': source_count,
                'confidence': confidence,
                'has_sources': has_sources,
                'answer_preview': answer[:100]
            })
            
            print(f"📤 Answer preview: {answer[:100]}...")
            print(f"📊 Sources: {source_count}, Confidence: {confidence:.3f}")
            print(f"🎯 Valid answer: {'❌' if is_not_specified else '✅'}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question': question,
                'is_not_specified': True,
                'source_count': 0,
                'confidence': 0,
                'has_sources': False,
                'answer_preview': f"Error: {e}"
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    total_questions = len(results)
    not_specified_count = sum(1 for r in results if r['is_not_specified'])
    valid_answers = total_questions - not_specified_count
    
    print(f"Total questions tested: {total_questions}")
    print(f"Valid answers: {valid_answers}")
    print(f"'Manual does not specify' responses: {not_specified_count}")
    print(f"Success rate: {(valid_answers/total_questions)*100:.1f}%")
    
    if not_specified_count > 0:
        print(f"\n❌ FAILED QUESTIONS:")
        for r in results:
            if r['is_not_specified']:
                print(f"  - '{r['question']}'")
                print(f"    Sources: {r['source_count']}, Confidence: {r['confidence']:.3f}")
    
    if valid_answers > 0:
        print(f"\n✅ SUCCESSFUL QUESTIONS:")
        for r in results:
            if not r['is_not_specified']:
                print(f"  - '{r['question']}'")
                print(f"    Sources: {r['source_count']}, Confidence: {r['confidence']:.3f}")
    
    return results

if __name__ == "__main__":
    test_questions()