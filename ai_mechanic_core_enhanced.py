import os
import json
import requests
from typing import List, Dict, Tuple, Optional
import openai
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from enum import Enum
import time

# Load environment variables
load_dotenv()

class SourceType(Enum):
    TRACKMEAI_DOCS = "trackmeai_docs"
    INTERNET_SEARCH = "internet_search"
    GENERAL_AI = "general_ai"

@dataclass
class SourceChunk:
    content: str
    source_type: SourceType
    confidence: float
    metadata: Dict
    url: Optional[str] = None

@dataclass
class EnhancedResponse:
    answer: str
    source_chunks: List[SourceChunk]
    total_confidence: float
    has_trackmeai_sources: bool
    has_internet_sources: bool

class InternetSearchProvider:
    """Internet search integration for automotive information"""
    
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        
        # Automotive-specific domains for better results
        self.automotive_domains = [
            "mechanicbase.com",
            "autozone.com", 
            "repairpal.com",
            "yourmechanic.com",
            "carcomplaints.com",
            "edmunds.com",
            "carfax.com",
            "kelleybluebook.com",
            "motortrend.com",
            "caranddriver.com",
            "reddit.com/r/MechanicAdvice",
            "reddit.com/r/Cars",
            "alldata.com",
            "mitchell1.com"
        ]
    
    def search_automotive_content(self, query: str, max_results: int = 5) -> List[SourceChunk]:
        """Search for automotive content on the internet"""
        try:
            if self.tavily_api_key:
                return self._search_with_tavily(query, max_results)
            elif self.serper_api_key:
                return self._search_with_serper(query, max_results)
            else:
                print("No internet search API configured")
                return []
        except Exception as e:
            print(f"Internet search error: {e}")
            return []
    
    def _search_with_tavily(self, query: str, max_results: int) -> List[SourceChunk]:
        """Search using Tavily API"""
        url = "https://api.tavily.com/search"
        
        # Enhanced query for automotive context
        automotive_query = f"automotive mechanic car repair {query}"
        
        payload = {
            "api_key": self.tavily_api_key,
            "query": automotive_query,
            "search_depth": "advanced",
            "include_domains": self.automotive_domains,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            return [
                SourceChunk(
                    content=result.get("content", "")[:1000],  # Limit content length
                    source_type=SourceType.INTERNET_SEARCH,
                    confidence=result.get("score", 0.5),
                    metadata={
                        "title": result.get("title", ""),
                        "domain": result.get("url", "").split("/")[2] if result.get("url") else ""
                    },
                    url=result.get("url")
                )
                for result in results if result.get("content")
            ]
        
        return []
    
    def _search_with_serper(self, query: str, max_results: int) -> List[SourceChunk]:
        """Search using Serper API"""
        url = "https://google.serper.dev/search"
        
        automotive_query = f"automotive mechanic car repair {query}"
        
        payload = {
            "q": automotive_query,
            "num": max_results,
            "gl": "us",
            "hl": "en"
        }
        
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            results = response.json().get("organic", [])
            return [
                SourceChunk(
                    content=result.get("snippet", ""),
                    source_type=SourceType.INTERNET_SEARCH,
                    confidence=0.6,  # Default confidence for Serper results
                    metadata={
                        "title": result.get("title", ""),
                        "domain": result.get("link", "").split("/")[2] if result.get("link") else ""
                    },
                    url=result.get("link")
                )
                for result in results if result.get("snippet")
            ]
        
        return []

class EnhancedAIMechanicCore:
    """Enhanced AI Mechanic with dual-source RAG capability"""
    
    def __init__(self, embeddings_file: str = "chunks_with_embeddings_enhanced_ocr.json"):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Load TrackMeAI embeddings data
        self.load_embeddings(embeddings_file)
        
        # Initialize internet search provider
        self.search_provider = InternetSearchProvider()
        
        # Configuration - Focus on PDF documentation only
        self.trackmeai_confidence_threshold = 0.15  # Threshold for considering sources relevant
        self.use_internet_fallback = False  # Disable internet search
        self.max_internet_results = 0  # No internet results
        self.always_search_internet_for_non_radical = False  # No internet search
        
    def load_embeddings(self, embeddings_file: str):
        """Load the TrackMeAI embeddings data from file"""
        try:
            with open(embeddings_file, "r") as f:
                self.documents = json.load(f)
            
            # Convert embeddings to numpy arrays for faster similarity search
            self.embeddings_matrix = np.array([doc["embedding_vector"] for doc in self.documents])
            
            print(f"Loaded {len(self.documents)} TrackMeAI documents with embeddings")
            print(f"Embedding dimensions: {self.embeddings_matrix.shape[1]}")
            
        except FileNotFoundError:
            print(f"TrackMeAI embeddings file {embeddings_file} not found. Operating with internet search only.")
            self.documents = []
            self.embeddings_matrix = np.array([])
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
    
    def search_trackmeai_docs(self, query: str, top_k: int = 8) -> List[SourceChunk]:
        """Search TrackMeAI proprietary documents with improved retrieval"""
        if len(self.documents) == 0:
            print("❌ No documents loaded for search")
            return []
        
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Generate and normalize query embedding
            query_embedding = self.generate_query_embedding(query)
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize document embeddings if not already normalized
            doc_embeddings_normalized = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            
            # Calculate cosine similarities with normalized embeddings
            similarities = np.dot(doc_embeddings_normalized, query_embedding_normalized)
            
            # Get top k most similar documents (already sorted by np.argsort)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            print(f"📊 Retrieved {len(top_indices)} chunks (top {top_k}):")
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                doc = self.documents[idx]
                similarity_score = float(similarities[idx])
                
                # Log retrieved chunks for debugging
                print(f"  {rank}. Similarity: {similarity_score:.3f} | Section: {doc['section_title'][:50]}...")
                print(f"     Content: {doc['chunk_text'][:100]}...")
                
                # Apply minimum similarity threshold
                if similarity_score > 0.1:  # Keep threshold low for inclusivity
                    results.append(
                        SourceChunk(
                            content=doc["chunk_text"],
                            source_type=SourceType.TRACKMEAI_DOCS,
                            confidence=similarity_score,
                            metadata={
                                "manual_id": doc["manual_id"],
                                "section_title": doc["section_title"],
                                "page_start": doc.get("page_start"),
                                "chunk_index": doc.get("chunk_index"),
                                "token_count": doc.get("token_count", 0),
                                "rank": rank
                            }
                        )
                    )
                else:
                    print(f"     ⚠️ Below threshold ({similarity_score:.3f} < 0.1)")
            
            print(f"✅ Returning {len(results)} chunks above threshold")
            return results
            
        except Exception as e:
            print(f"❌ Error searching TrackMeAI docs: {e}")
            return []
    
    def is_radical_related_question(self, query: str) -> bool:
        """Detect if question is about Radical cars or related to our documents"""
        radical_keywords = [
            'radical', 'sr3', 'hankook', 'dunlop', 'track', 'racing',
            'aerodynamics', 'roll cage', 'harness', 'racing seat'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in radical_keywords)
    
    def evaluate_source_confidence(self, trackmeai_sources: List[SourceChunk]) -> float:
        """Evaluate if TrackMeAI sources provide sufficient confidence"""
        if not trackmeai_sources:
            return 0.0
        
        # Use highest confidence score from TrackMeAI sources
        max_confidence = max(source.confidence for source in trackmeai_sources)
        return max_confidence
    
    def merge_short_chunks(self, sources: List[SourceChunk]) -> List[SourceChunk]:
        """Merge short chunks with next chunk for better context"""
        if not sources:
            return sources
        
        merged_sources = []
        i = 0
        
        while i < len(sources):
            current_chunk = sources[i]
            current_tokens = current_chunk.metadata.get("token_count", len(current_chunk.content.split()) * 1.3)
            
            # If chunk is short and there's a next chunk, try to merge
            if current_tokens < 100 and i + 1 < len(sources):
                next_chunk = sources[i + 1]
                next_tokens = next_chunk.metadata.get("token_count", len(next_chunk.content.split()) * 1.3)
                
                # Only merge if combined size is reasonable
                if current_tokens + next_tokens < 500:
                    print(f"🔗 Merging short chunks: {current_tokens} + {next_tokens} tokens")
                    
                    # Create merged chunk
                    merged_content = f"{current_chunk.content}\n\n---\n\n{next_chunk.content}"
                    merged_metadata = current_chunk.metadata.copy()
                    merged_metadata["merged_with_next"] = True
                    merged_metadata["token_count"] = current_tokens + next_tokens
                    
                    merged_chunk = SourceChunk(
                        content=merged_content,
                        source_type=current_chunk.source_type,
                        confidence=max(current_chunk.confidence, next_chunk.confidence),
                        metadata=merged_metadata
                    )
                    
                    merged_sources.append(merged_chunk)
                    i += 2  # Skip next chunk since we merged it
                    continue
            
            merged_sources.append(current_chunk)
            i += 1
        
        return merged_sources
    
    def build_context_prompt(self, sources: List[SourceChunk], max_tokens: int = 3500) -> str:
        """Build context with token limit management"""
        if not sources:
            return ""
        
        context_parts = []
        total_tokens = 0
        
        print(f"📝 Building context from {len(sources)} sources (max {max_tokens} tokens):")
        
        for i, source in enumerate(sources, 1):
            # Estimate tokens for this chunk
            chunk_tokens = source.metadata.get("token_count", len(source.content.split()) * 1.3)
            section_header = f"[CHUNK {i} - {source.metadata.get('section_title', 'Unknown')}]\n{source.content}\n\n---\n\n"
            header_tokens = len(section_header.split()) * 1.3
            
            if total_tokens + header_tokens > max_tokens:
                print(f"  ⚠️ Token limit reached. Using {i-1}/{len(sources)} chunks")
                break
            
            context_parts.append(section_header)
            total_tokens += header_tokens
            print(f"  {i}. Added {chunk_tokens:.0f} tokens from '{source.metadata.get('section_title', 'Unknown')[:30]}...'")
        
        print(f"✅ Context built: {total_tokens:.0f} total tokens")
        return "\n".join(context_parts)
    
    def synthesize_response(self, 
                          query: str, 
                          trackmeai_sources: List[SourceChunk], 
                          internet_sources: List[SourceChunk]) -> EnhancedResponse:
        """Synthesize response using improved grounded prompting"""
        
        print(f"🤖 Synthesizing response for: '{query}'")
        
        # Merge short chunks for better context
        merged_sources = self.merge_short_chunks(trackmeai_sources)
        
        # Check if we have good matches
        best_similarity = max([s.confidence for s in merged_sources]) if merged_sources else 0
        print(f"📊 Best similarity score: {best_similarity:.3f}")
        
        # If no good matches, return "not specified" response
        if not merged_sources or best_similarity < 0.15:
            print("⚠️ No relevant chunks found above threshold")
            return EnhancedResponse(
                answer="The manual does not specify this information. Please consult the complete Radical SR3 Owner's Manual or contact Radical support for detailed guidance.",
                source_chunks=[],
                total_confidence=0.2,
                has_trackmeai_sources=False,
                has_internet_sources=False
            )
        
        # Build context with token management
        context = self.build_context_prompt(merged_sources, max_tokens=3000)
        
        if not context.strip():
            return EnhancedResponse(
                answer="The manual does not specify this information.",
                source_chunks=[],
                total_confidence=0.2,
                has_trackmeai_sources=False,
                has_internet_sources=False
            )
        
        # Improved grounded prompting
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are the Radical Assistant for Radical SR3 race cars. SPECIAL BEHAVIORS: 1) PURCHASE FLOW: When user wants to buy/purchase/order parts (tires, brake pads, oil), follow: detect part → ask options (Dunlop/Hankook?) → confirm mock payment → ask delivery address → ask 'spare inventory or install on car?' → confirm completion. 2) CONVERSATIONAL: Keep responses short and conversational, avoid technical dumps. 3) FOLLOW-UP: After answering, ask 'Would you like me to retrieve any further information related to this?' when relevant. Use ONLY provided manual sections. If not covered, say 'The manual does not specify this information.'"
                    },
                    {
                        "role": "system", 
                        "content": f"Relevant manual sections:\n\n{context}"
                    },
                    {
                        "role": "user", 
                        "content": query
                    }
                ],
                temperature=0.05,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on similarity scores
            if best_similarity >= 0.4:
                total_confidence = 0.95
            elif best_similarity >= 0.3:
                total_confidence = 0.85
            elif best_similarity >= 0.2:
                total_confidence = 0.75
            else:
                total_confidence = 0.65
            
            print(f"✅ Response generated with {total_confidence:.2f} confidence")
            
            return EnhancedResponse(
                answer=answer,
                source_chunks=merged_sources,
                total_confidence=total_confidence,
                has_trackmeai_sources=len(merged_sources) > 0,
                has_internet_sources=False
            )
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            raise Exception(f"Error generating response: {e}")
    
    def detect_purchase_intent(self, question: str) -> Dict:
        """Detect if user wants to purchase parts"""
        purchase_keywords = ['buy', 'purchase', 'order', 'get', 'need']
        part_keywords = {
            'tires': ['tire', 'tires', 'tyre', 'tyres'],
            'brake_pads': ['brake pad', 'brake pads', 'pads'],
            'oil': ['oil', 'engine oil', 'motor oil'],
            'filters': ['filter', 'air filter', 'oil filter'],
            'spark_plugs': ['spark plug', 'spark plugs', 'plugs']
        }
        
        question_lower = question.lower()
        
        # Check for purchase intent
        has_purchase_intent = any(keyword in question_lower for keyword in purchase_keywords)
        
        if has_purchase_intent:
            # Detect which part
            for part_type, keywords in part_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    return {
                        "has_intent": True,
                        "part_type": part_type,
                        "original_question": question
                    }
        
        return {"has_intent": False}
    
    def handle_purchase_flow(self, part_type: str, question: str) -> Dict:
        """Handle mock purchase flow"""
        part_options = {
            'tires': ['Dunlop Racing Slicks', 'Hankook Z207 Compound'],
            'brake_pads': ['High-Performance Racing Pads', 'Endurance Track Pads'],
            'oil': ['Radical SR3 Engine Oil (5W-30)', 'Racing Grade Synthetic Oil'],
            'filters': ['OEM Air Filter', 'High-Flow Performance Filter'],
            'spark_plugs': ['NGK Racing Plugs', 'Denso Iridium Plugs']
        }
        
        options = part_options.get(part_type, [f"{part_type.replace('_', ' ').title()} Option A", f"{part_type.replace('_', ' ').title()} Option B"])
        
        return {
            "answer": f"I can help you order {part_type.replace('_', ' ')}! \n\nWhich option would you prefer:\n1. {options[0]}\n2. {options[1]}\n\nPlease let me know your choice and I'll process the order.",
            "source_chunks": [],
            "question": question,
            "top_k": 0,
            "total_confidence": 1.0,
            "has_trackmeai_sources": False,
            "has_internet_sources": False,
            "error": False,
            "purchase_flow": True,
            "purchase_step": "selection",
            "part_type": part_type,
            "options": options
        }
    
    def handle_purchase_selection(self, selection: str, part_type: str, options: list) -> Dict:
        """Handle part selection in purchase flow"""
        try:
            choice_num = int(selection) - 1
            if 0 <= choice_num < len(options):
                selected_part = options[choice_num]
                mock_price = {"tires": "$450", "brake_pads": "$180", "oil": "$85", "filters": "$45", "spark_plugs": "$95"}.get(part_type, "$100")
                
                return {
                    "answer": f"Great choice! You've selected: **{selected_part}**\n\nPrice: {mock_price}\n\n💳 **Payment Confirmed** (simulated)\n\nPlease provide your delivery address so I can arrange shipping.",
                    "source_chunks": [],
                    "question": f"Selected {selected_part}",
                    "top_k": 0,
                    "total_confidence": 1.0,
                    "has_trackmeai_sources": False,
                    "has_internet_sources": False,
                    "error": False,
                    "purchase_flow": True,
                    "purchase_step": "address",
                    "selected_part": selected_part,
                    "part_type": part_type,
                    "price": mock_price
                }
        except:
            pass
        
        return {
            "answer": "Please select a valid option (1 or 2) from the list above.",
            "source_chunks": [],
            "purchase_flow": True,
            "purchase_step": "selection",
            "error": False
        }
    
    def handle_delivery_address(self, address: str, selected_part: str, part_type: str) -> Dict:
        """Handle delivery address in purchase flow"""
        return {
            "answer": f"Perfect! Your order details:\n\n📦 **{selected_part}**\n📍 **Delivery to:** {address}\n\n✅ **Order Confirmed!**\n\n👉 **Final question:** Do you want to add this part to your spare inventory or directly install it onto the car?\n\nType 'inventory' or 'install' to complete your order.",
            "source_chunks": [],
            "question": f"Delivery to {address}",
            "top_k": 0,
            "total_confidence": 1.0,
            "has_trackmeai_sources": False,
            "has_internet_sources": False,
            "error": False,
            "purchase_flow": True,
            "purchase_step": "final_choice",
            "selected_part": selected_part,
            "part_type": part_type,
            "delivery_address": address
        }
    
    def handle_final_choice(self, choice: str, selected_part: str, delivery_address: str) -> Dict:
        """Handle final inventory vs install choice"""
        choice_lower = choice.lower()
        
        if 'inventory' in choice_lower or 'spare' in choice_lower:
            action = "added to your spare parts inventory"
            icon = "📦"
        elif 'install' in choice_lower or 'car' in choice_lower:
            action = "scheduled for direct installation on your car"
            icon = "🔧"
        else:
            return {
                "answer": "Please specify: 'inventory' (add to spare parts) or 'install' (direct installation on car)",
                "purchase_flow": True,
                "purchase_step": "final_choice",
                "error": False
            }
        
        return {
            "answer": f"{icon} **Order Complete!**\n\n✅ Your **{selected_part}** will be {action}\n📍 Delivery to: {delivery_address}\n🚚 Expected delivery: 2-3 business days\n\nThank you for your order! You'll receive tracking information via email.",
            "source_chunks": [],
            "question": f"Final choice: {choice}",
            "top_k": 0,
            "total_confidence": 1.0,
            "has_trackmeai_sources": False,
            "has_internet_sources": False,
            "error": False,
            "purchase_flow": True,
            "purchase_step": "completed",
            "final_action": action
        }
    
    def ask_ai_mechanic(self, question: str, top_k: int = 5) -> Dict:
        """Enhanced AI mechanic with intelligent dual-source capability"""
        try:
            print(f"Processing question: {question}")
            
            # Step 0: Check for purchase intent first
            purchase_intent = self.detect_purchase_intent(question)
            if purchase_intent["has_intent"]:
                print(f"🛒 Purchase intent detected for: {purchase_intent['part_type']}")
                return self.handle_purchase_flow(purchase_intent["part_type"], question)
            
            # Step 1: Always search TrackMeAI documents first (get top 8)
            trackmeai_sources = self.search_trackmeai_docs(question, top_k=8)
            print(f"📋 Retrieved {len(trackmeai_sources)} manual sources")
            
            # Step 2: Evaluate manual source confidence
            trackmeai_confidence = self.evaluate_source_confidence(trackmeai_sources)
            print(f"Manual source confidence: {trackmeai_confidence:.3f}")
            
            # No internet sources - PDF documentation only
            internet_sources = []
            
            # Use all relevant manual sources
            filtered_trackmeai_sources = trackmeai_sources
            
            # Step 3: Synthesize response from manual sources
            enhanced_response = self.synthesize_response(question, filtered_trackmeai_sources, internet_sources)
            
            # Step 4: Format response for API compatibility with enhanced metadata
            formatted_sources = []
            for source in enhanced_response.source_chunks:
                formatted_source = {
                    "content": source.content,
                    "source_type": source.source_type.value,
                    "confidence": source.confidence,
                    "metadata": {
                        **source.metadata,
                        "source_display": f"From {source.metadata.get('section_title', 'Unknown Section')} of Radical SR3 Manual",
                        "page_info": f"Page {source.metadata.get('page_start', 'Unknown')}" if source.metadata.get('page_start') else "Page Unknown"
                    }
                }
                if source.url:
                    formatted_source["url"] = source.url
                formatted_sources.append(formatted_source)
            
            return {
                "answer": enhanced_response.answer,
                "source_chunks": formatted_sources,
                "question": question,
                "top_k": top_k,
                "total_confidence": enhanced_response.total_confidence,
                "has_trackmeai_sources": enhanced_response.has_trackmeai_sources,
                "has_internet_sources": enhanced_response.has_internet_sources,
                "error": False
            }
            
        except Exception as e:
            print(f"Error in ask_ai_mechanic: {e}")
            # Even on error, provide a helpful response using GPT-4 directly
            try:
                fallback_response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert automotive mechanic. Provide helpful advice for this automotive question."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                return {
                    "answer": fallback_response.choices[0].message.content,
                    "source_chunks": [],
                    "question": question,
                    "top_k": top_k,
                    "total_confidence": 0.6,
                    "has_trackmeai_sources": False,
                    "has_internet_sources": False,
                    "error": False
                }
            except:
                return {
                    "answer": "I apologize, but I'm having technical difficulties right now. For automotive assistance, I recommend consulting a qualified mechanic or checking trusted automotive resources online.",
                    "source_chunks": [],
                    "question": question,
                    "top_k": top_k,
                    "total_confidence": 0.0,
                    "has_trackmeai_sources": False,
                    "has_internet_sources": False,
                    "error": True
                }
    
    def get_available_manuals(self) -> List[str]:
        """Get list of available TrackMeAI manual IDs"""
        if not self.documents:
            return []
        
        manual_ids = set()
        for doc in self.documents:
            manual_ids.add(doc["manual_id"])
        
        return list(manual_ids)
    
    def get_manual_sections(self, manual_id: str) -> List[str]:
        """Get sections for a specific manual"""
        sections = set()
        for doc in self.documents:
            if doc["manual_id"] == manual_id:
                sections.add(doc["section_title"])
        
        return list(sections)

# Backward compatibility - create instance with original class name
AIMechanicCore = EnhancedAIMechanicCore