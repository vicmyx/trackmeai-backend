#!/usr/bin/env python3
"""
Debug script for AI Mechanic system to identify why it returns 
"The manual does not specify this information" for all questions.
"""

import os
import json
import numpy as np
from ai_mechanic_core_enhanced import EnhancedAIMechanicCore

def run_diagnostics():
    """Run comprehensive diagnostics on the AI mechanic system"""
    
    print("🔧 AI MECHANIC DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Test 1: Check file existence and basic loading
    print("\n1. FILE SYSTEM CHECK")
    print("-" * 20)
    
    embeddings_file = "chunks_with_embeddings_advanced.json"
    if os.path.exists(embeddings_file):
        print(f"✅ {embeddings_file} exists")
        
        # Check file size
        file_size = os.path.getsize(embeddings_file)
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        # Try to load and examine structure
        try:
            with open(embeddings_file, 'r') as f:
                data = json.load(f)
            
            print(f"📄 Documents loaded: {len(data)}")
            
            if len(data) > 0:
                sample_doc = data[0]
                print(f"📋 Sample document keys: {list(sample_doc.keys())}")
                
                if 'embedding_vector' in sample_doc:
                    embedding_length = len(sample_doc['embedding_vector'])
                    print(f"🧮 Embedding dimensions: {embedding_length}")
                    
                    # Check embedding values
                    embedding_sample = sample_doc['embedding_vector'][:5]
                    print(f"📈 Sample embedding values: {embedding_sample}")
                    
                    # Check if embeddings are all zeros or have valid values
                    embedding_array = np.array(sample_doc['embedding_vector'])
                    print(f"📊 Embedding stats - Min: {embedding_array.min():.6f}, Max: {embedding_array.max():.6f}, Mean: {embedding_array.mean():.6f}")
                    
                    if np.all(embedding_array == 0):
                        print("❌ WARNING: Embedding vector is all zeros!")
                    else:
                        print("✅ Embedding vector has valid values")
                else:
                    print("❌ No 'embedding_vector' field found in document!")
                
                print(f"📑 Sample section title: {sample_doc.get('section_title', 'N/A')}")
                print(f"📄 Sample text (first 100 chars): {sample_doc.get('chunk_text', 'N/A')[:100]}...")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
    else:
        print(f"❌ {embeddings_file} NOT FOUND!")
        return
    
    # Test 2: Initialize AI Mechanic
    print("\n2. AI MECHANIC INITIALIZATION")
    print("-" * 30)
    
    try:
        ai_mechanic = EnhancedAIMechanicCore(embeddings_file)
        print("✅ EnhancedAIMechanicCore initialized successfully")
        
        print(f"📊 Documents loaded: {len(ai_mechanic.documents)}")
        print(f"🧮 Embeddings matrix shape: {ai_mechanic.embeddings_matrix.shape}")
        
        # Check for duplicate embeddings
        unique_embeddings = np.unique(ai_mechanic.embeddings_matrix, axis=0)
        if len(unique_embeddings) < len(ai_mechanic.embeddings_matrix):
            print(f"⚠️ WARNING: Found duplicate embeddings! {len(ai_mechanic.embeddings_matrix)} total, {len(unique_embeddings)} unique")
        else:
            print("✅ All embeddings are unique")
        
    except Exception as e:
        print(f"❌ Failed to initialize AI Mechanic: {e}")
        return
    
    # Test 3: Test query embedding generation
    print("\n3. QUERY EMBEDDING TEST")
    print("-" * 25)
    
    test_query = "What is the pre-race checklist?"
    print(f"🔍 Test query: '{test_query}'")
    
    try:
        query_embedding = ai_mechanic.generate_query_embedding(test_query)
        print(f"✅ Query embedding generated successfully")
        print(f"🧮 Query embedding shape: {query_embedding.shape}")
        print(f"📊 Query embedding stats - Min: {query_embedding.min():.6f}, Max: {query_embedding.max():.6f}, Mean: {query_embedding.mean():.6f}")
        
        if np.all(query_embedding == 0):
            print("❌ WARNING: Query embedding is all zeros!")
        else:
            print("✅ Query embedding has valid values")
            
    except Exception as e:
        print(f"❌ Failed to generate query embedding: {e}")
        return
    
    # Test 4: Test similarity calculations
    print("\n4. SIMILARITY CALCULATION TEST")
    print("-" * 32)
    
    try:
        # Manual similarity calculation for debugging
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings_normalized = ai_mechanic.embeddings_matrix / np.linalg.norm(ai_mechanic.embeddings_matrix, axis=1, keepdims=True)
        similarities = np.dot(doc_embeddings_normalized, query_embedding_normalized)
        
        print(f"📊 Similarity scores - Min: {similarities.min():.6f}, Max: {similarities.max():.6f}, Mean: {similarities.mean():.6f}")
        
        # Show top 5 similarities
        top_5_indices = np.argsort(similarities)[::-1][:5]
        print("\n🔝 Top 5 similarity scores:")
        for i, idx in enumerate(top_5_indices, 1):
            doc = ai_mechanic.documents[idx]
            similarity = similarities[idx]
            print(f"  {i}. Score: {similarity:.6f} | Section: {doc['section_title'][:50]}...")
            
        # Check threshold
        threshold = 0.15
        above_threshold = np.sum(similarities > threshold)
        print(f"\n📊 Documents above threshold ({threshold}): {above_threshold}/{len(similarities)}")
        
        if above_threshold == 0:
            print("❌ WARNING: No documents above similarity threshold!")
            print("💡 This could be why you're getting 'manual does not specify' responses")
            
            # Try lower thresholds
            for test_threshold in [0.1, 0.05, 0.01]:
                above_test = np.sum(similarities > test_threshold)
                print(f"   📊 Above {test_threshold}: {above_test} documents")
        else:
            print("✅ Some documents are above threshold")
            
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")
        return
    
    # Test 5: Full search test
    print("\n5. FULL SEARCH TEST")
    print("-" * 20)
    
    try:
        search_results = ai_mechanic.search_trackmeai_docs(test_query, top_k=5)
        print(f"📋 Search returned {len(search_results)} results")
        
        if len(search_results) == 0:
            print("❌ No search results returned!")
        else:
            print("\n📑 Search results:")
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. Confidence: {result.confidence:.6f}")
                print(f"     Section: {result.metadata.get('section_title', 'Unknown')}")
                print(f"     Content preview: {result.content[:100]}...")
                print()
    except Exception as e:
        print(f"❌ Error in search test: {e}")
        return
    
    # Test 6: Full AI mechanic question
    print("\n6. FULL AI MECHANIC TEST")
    print("-" * 25)
    
    try:
        response = ai_mechanic.ask_ai_mechanic(test_query)
        
        print(f"📤 Response received")
        print(f"✅ Answer: {response['answer'][:200]}...")
        print(f"📊 Source chunks: {len(response['source_chunks'])}")
        print(f"🎯 Confidence: {response['total_confidence']}")
        print(f"📋 Has TrackMeAI sources: {response['has_trackmeai_sources']}")
        
        if "manual does not specify" in response['answer'].lower():
            print("❌ ISSUE FOUND: Returning 'manual does not specify' response!")
        else:
            print("✅ Returned valid answer from manual")
            
    except Exception as e:
        print(f"❌ Error in full AI mechanic test: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🔧 DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    run_diagnostics()