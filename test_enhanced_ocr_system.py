#!/usr/bin/env python3
"""
Test script for the enhanced AI mechanic system with enhanced OCR embeddings.
Tests specific tire pressure questions and pre-race checklist.
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

def print_separator(title):
    """Print a nice separator with title"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)

def print_source_analysis(source_chunks):
    """Print detailed analysis of source chunks"""
    if not source_chunks:
        print("❌ No sources found")
        return
    
    print(f"📊 Found {len(source_chunks)} source chunks:")
    for i, source in enumerate(source_chunks, 1):
        print(f"\n  {i}. Similarity: {source['confidence']:.4f}")
        print(f"     Section: {source['metadata'].get('section_title', 'Unknown')}")
        print(f"     Page: {source['metadata'].get('page_start', 'Unknown')}")
        print(f"     Content preview: {source['content'][:150]}...")

def test_question(mechanic, question, expected_info=None):
    """Test a single question and analyze results"""
    print_separator(f"Testing: {question}")
    
    try:
        # Get the response
        result = mechanic.ask_ai_mechanic(question)
        
        print(f"🤖 Answer:")
        print(f"   {result['answer']}")
        print(f"\n📈 Confidence: {result['total_confidence']:.2f}")
        
        # Analyze sources
        print_source_analysis(result['source_chunks'])
        
        # Check if expected information is mentioned
        if expected_info:
            answer_lower = result['answer'].lower()
            found_info = []
            missing_info = []
            
            for info in expected_info:
                if info.lower() in answer_lower:
                    found_info.append(info)
                else:
                    missing_info.append(info)
            
            if found_info:
                print(f"\n✅ Found expected info: {', '.join(found_info)}")
            if missing_info:
                print(f"❌ Missing expected info: {', '.join(missing_info)}")
        
        # Return results for summary
        return {
            'question': question,
            'answer': result['answer'],
            'confidence': result['total_confidence'],
            'num_sources': len(result['source_chunks']),
            'max_similarity': max([s['confidence'] for s in result['source_chunks']]) if result['source_chunks'] else 0,
            'found_info': found_info if expected_info else [],
            'missing_info': missing_info if expected_info else []
        }
        
    except Exception as e:
        print(f"❌ Error testing question: {e}")
        return {
            'question': question,
            'error': str(e)
        }

def main():
    """Main test function"""
    print_separator("Enhanced OCR System Test")
    print("Testing enhanced AI mechanic with enhanced OCR embeddings")
    print("Expected tire pressure specs: Hankook 28psi, Dunlop 30psi hot")
    
    # Initialize the enhanced system
    try:
        mechanic = EnhancedAIMechanicCore(embeddings_file="chunks_with_embeddings_enhanced_ocr.json")
        print(f"✅ System initialized with {len(mechanic.documents)} documents")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test questions with expected information
    test_cases = [
        {
            'question': "What tire pressure should I use for Hankook tires?",
            'expected': ["28psi", "Hankook"]
        },
        {
            'question': "What is the correct tire pressure for Dunlop tires?",
            'expected': ["30psi", "Dunlop", "hot"]
        },
        {
            'question': "What is the pre-race checklist?",
            'expected': ["checklist", "inspect", "check"]
        }
    ]
    
    # Run tests
    results = []
    for test_case in test_cases:
        result = test_question(mechanic, test_case['question'], test_case['expected'])
        results.append(result)
    
    # Print summary
    print_separator("TEST SUMMARY")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['question']}: ERROR - {result['error']}")
        else:
            print(f"\n🔍 Question: {result['question']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {result['num_sources']}")
            print(f"   Max Similarity: {result['max_similarity']:.4f}")
            
            if result['found_info']:
                print(f"   ✅ Found: {', '.join(result['found_info'])}")
            if result['missing_info']:
                print(f"   ❌ Missing: {', '.join(result['missing_info'])}")
    
    # Overall assessment
    print_separator("ASSESSMENT")
    
    successful_tests = [r for r in results if 'error' not in r]
    high_confidence_tests = [r for r in successful_tests if r['confidence'] >= 0.75]
    good_similarity_tests = [r for r in successful_tests if r['max_similarity'] >= 0.3]
    
    print(f"📊 Results Summary:")
    print(f"   • Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"   • High confidence (≥0.75): {len(high_confidence_tests)}/{len(successful_tests)}")
    print(f"   • Good similarity (≥0.30): {len(good_similarity_tests)}/{len(successful_tests)}")
    
    # Check if tire pressure specs were found
    tire_pressure_found = False
    for result in results:
        if 'error' not in result and any('28psi' in info or '30psi' in info for info in result.get('found_info', [])):
            tire_pressure_found = True
            break
    
    if tire_pressure_found:
        print(f"   ✅ Tire pressure specifications successfully extracted!")
    else:
        print(f"   ❌ Tire pressure specifications not found in responses")

if __name__ == "__main__":
    main()