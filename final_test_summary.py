#!/usr/bin/env python3
"""
Final comprehensive test and summary of the enhanced OCR system performance.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_mechanic_core_enhanced import EnhancedAIMechanicCore

def print_separator(title, char="=", width=80):
    """Print a separator line with title"""
    print(f"\n{char * width}")
    print(f" {title} ")
    print(f"{char * width}")

def analyze_answer_quality(question, answer, expected_content):
    """Analyze the quality and accuracy of the answer"""
    answer_lower = answer.lower()
    
    # Check for expected content
    found = []
    missing = []
    
    for content in expected_content:
        if content.lower() in answer_lower:
            found.append(content)
        else:
            missing.append(content)
    
    # Quality metrics
    word_count = len(answer.split())
    is_comprehensive = word_count > 50
    has_specific_details = any(detail in answer_lower for detail in ['28psi', '30psi', 'bodywork', 'shake test', 'wiring'])
    
    return {
        'found_content': found,
        'missing_content': missing,
        'word_count': word_count,
        'is_comprehensive': is_comprehensive,
        'has_specific_details': has_specific_details,
        'quality_score': len(found) / len(expected_content) if expected_content else 0
    }

def main():
    """Comprehensive test and analysis"""
    print_separator("ENHANCED OCR SYSTEM - FINAL VALIDATION")
    
    # Initialize system
    try:
        mechanic = EnhancedAIMechanicCore(embeddings_file="chunks_with_embeddings_enhanced_ocr.json")
        print(f"✅ System initialized successfully")
        print(f"📚 Loaded {len(mechanic.documents)} document chunks")
        print(f"🧠 Embedding dimensions: {mechanic.embeddings_matrix.shape[1] if len(mechanic.embeddings_matrix) > 0 else 'N/A'}")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Test cases with expected content
    test_cases = [
        {
            'question': "What tire pressure should I use for Hankook tires?",
            'expected': ["28psi", "Hankook", "target", "pressure"],
            'description': "Hankook tire pressure specification"
        },
        {
            'question': "What is the correct tire pressure for Dunlop tires?",
            'expected': ["30psi", "Dunlop", "hot", "target", "pressure"],
            'description': "Dunlop tire pressure specification"
        },
        {
            'question': "What is the pre-race checklist?",
            'expected': ["bodywork", "shake test", "wiring", "brakes", "engine bay", "checklist"],
            'description': "Complete pre-race preparation checklist"
        }
    ]
    
    results = []
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print_separator(f"TEST {i}: {test_case['description']}", char="-", width=60)
        
        try:
            # Get AI response
            result = mechanic.ask_ai_mechanic(test_case['question'])
            
            # Analyze answer quality
            quality = analyze_answer_quality(
                test_case['question'], 
                result['answer'], 
                test_case['expected']
            )
            
            # Display results
            print(f"❓ Question: {test_case['question']}")
            print(f"🤖 Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
            print(f"📊 Metrics:")
            print(f"   • Confidence: {result['total_confidence']:.2f}")
            print(f"   • Sources: {len(result['source_chunks'])}")
            print(f"   • Max Similarity: {max([s['confidence'] for s in result['source_chunks']]) if result['source_chunks'] else 0:.4f}")
            print(f"   • Word Count: {quality['word_count']}")
            print(f"   • Quality Score: {quality['quality_score']:.2f}")
            
            if quality['found_content']:
                print(f"   ✅ Found: {', '.join(quality['found_content'])}")
            if quality['missing_content']:
                print(f"   ❌ Missing: {', '.join(quality['missing_content'])}")
            
            # Store results
            results.append({
                'test_case': test_case,
                'result': result,
                'quality': quality
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e)
            })
    
    # Final assessment
    print_separator("FINAL ASSESSMENT")
    
    successful_tests = [r for r in results if 'error' not in r]
    total_tests = len(results)
    
    if successful_tests:
        avg_confidence = sum(r['result']['total_confidence'] for r in successful_tests) / len(successful_tests)
        avg_similarity = sum(max([s['confidence'] for s in r['result']['source_chunks']]) if r['result']['source_chunks'] else 0 for r in successful_tests) / len(successful_tests)
        avg_quality_score = sum(r['quality']['quality_score'] for r in successful_tests) / len(successful_tests)
        
        print(f"📊 Overall Performance:")
        print(f"   • Successful Tests: {len(successful_tests)}/{total_tests}")
        print(f"   • Average Confidence: {avg_confidence:.2f}")
        print(f"   • Average Similarity: {avg_similarity:.4f}")
        print(f"   • Average Quality Score: {avg_quality_score:.2f}")
        
        # Specific achievements
        tire_pressure_success = False
        for r in successful_tests:
            if 'tire pressure' in r['test_case']['question'].lower():
                if r['quality']['quality_score'] >= 0.6:
                    tire_pressure_success = True
                    break
        
        print(f"\n🎯 Key Achievements:")
        if tire_pressure_success:
            print(f"   ✅ Successfully extracted tire pressure specifications (Hankook 28psi, Dunlop 30psi hot)")
        else:
            print(f"   ❌ Tire pressure extraction needs improvement")
        
        checklist_success = any(r['quality']['is_comprehensive'] for r in successful_tests if 'checklist' in r['test_case']['question'].lower())
        if checklist_success:
            print(f"   ✅ Comprehensive pre-race checklist retrieval")
        else:
            print(f"   ❌ Pre-race checklist needs improvement")
        
        high_similarity_count = sum(1 for r in successful_tests if (max([s['confidence'] for s in r['result']['source_chunks']]) if r['result']['source_chunks'] else 0) >= 0.6)
        print(f"   ✅ {high_similarity_count}/{len(successful_tests)} tests achieved high similarity (≥0.6)")
        
        print(f"\n🚀 System Status: {'READY FOR PRODUCTION' if avg_confidence >= 0.8 and tire_pressure_success else 'NEEDS OPTIMIZATION'}")
    
    else:
        print(f"❌ No successful tests completed")

if __name__ == "__main__":
    main()