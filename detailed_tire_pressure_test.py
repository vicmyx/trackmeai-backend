#!/usr/bin/env python3
"""
Detailed test to examine the tire pressure content retrieved from enhanced OCR system.
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

def print_detailed_source_content(source_chunks):
    """Print the full content of source chunks"""
    if not source_chunks:
        print("❌ No sources found")
        return
    
    for i, source in enumerate(source_chunks, 1):
        print(f"\n--- SOURCE CHUNK {i} ---")
        print(f"Similarity: {source['confidence']:.4f}")
        print(f"Section: {source['metadata'].get('section_title', 'Unknown')}")
        print(f"Page: {source['metadata'].get('page_start', 'Unknown')}")
        print(f"Full Content:")
        print(source['content'])
        print("-" * 50)

def main():
    """Test tire pressure questions in detail"""
    print("=" * 80)
    print(" DETAILED TIRE PRESSURE TEST ")
    print("=" * 80)
    
    # Initialize the enhanced system
    try:
        mechanic = EnhancedAIMechanicCore(embeddings_file="chunks_with_embeddings_enhanced_ocr.json")
        print(f"✅ System initialized with {len(mechanic.documents)} documents")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test Hankook tire pressure
    print("\n" + "="*60)
    print(" HANKOOK TIRE PRESSURE QUESTION ")
    print("="*60)
    
    question = "What tire pressure should I use for Hankook tires?"
    result = mechanic.ask_ai_mechanic(question)
    
    print(f"🤖 Answer: {result['answer']}")
    print(f"📈 Confidence: {result['total_confidence']:.2f}")
    print_detailed_source_content(result['source_chunks'])
    
    # Test Dunlop tire pressure
    print("\n" + "="*60)
    print(" DUNLOP TIRE PRESSURE QUESTION ")
    print("="*60)
    
    question = "What is the correct tire pressure for Dunlop tires?"
    result = mechanic.ask_ai_mechanic(question)
    
    print(f"🤖 Answer: {result['answer']}")
    print(f"📈 Confidence: {result['total_confidence']:.2f}")
    print_detailed_source_content(result['source_chunks'])

if __name__ == "__main__":
    main()