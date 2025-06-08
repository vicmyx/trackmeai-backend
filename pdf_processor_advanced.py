import os
import requests
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict, Tuple
import re
import json
import openai
from dotenv import load_dotenv
import time
import numpy as np

# Load environment variables
load_dotenv()

class AdvancedPDFProcessor:
    def __init__(self, max_tokens_per_chunk: int = 400, overlap_tokens: int = 50):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        
        # Initialize OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def download_pdf(self, url: str, filename: str) -> str:
        """Download PDF from URL and save locally"""
        print(f"Downloading {filename}...")
        
        response = requests.get(url)
        response.raise_for_status()
        
        os.makedirs("pdfs", exist_ok=True)
        filepath = f"pdfs/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded: {filepath}")
        return filepath
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def extract_content_comprehensive(self, pdf_path: str) -> List[Dict]:
        """Extract all meaningful content from PDF"""
        doc = fitz.open(pdf_path)
        all_content = []
        
        print(f"Processing {doc.page_count} pages...")
        
        # Extract content from all pages
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text using the best method for this page
            text = self._extract_best_text(page)
            
            if not text or len(text.strip()) < 30:
                continue
            
            # Clean the text
            cleaned_text = self._clean_text_advanced(text)
            
            # Skip if still too short after cleaning
            if len(cleaned_text.strip()) < 50:
                continue
            
            # Count meaningful words
            words = [w for w in cleaned_text.split() if len(w) > 2 and w.isalpha()]
            
            if len(words) < 15:  # Need at least 15 meaningful words
                continue
            
            all_content.append({
                "page": page_num + 1,
                "content": cleaned_text.strip()
            })
        
        doc.close()
        
        # Combine all content into one large text for token-based chunking
        combined_content = ""
        page_markers = {}  # Track which part of text came from which page
        
        for page_data in all_content:
            start_pos = len(combined_content)
            page_content = f"\n\n[Page {page_data['page']}]\n{page_data['content']}"
            combined_content += page_content
            page_markers[page_data['page']] = (start_pos, len(combined_content))
        
        print(f"Extracted content from {len(all_content)} pages")
        print(f"Total content length: {len(combined_content)} characters")
        print(f"Total tokens: {self.count_tokens(combined_content)}")
        
        return combined_content, page_markers
    
    def _extract_best_text(self, page) -> str:
        """Try multiple extraction methods and return the best result"""
        methods = [
            page.get_text,
            lambda: self._extract_from_dict(page),
            lambda: self._extract_from_blocks(page)
        ]
        
        best_text = ""
        best_word_count = 0
        
        for method in methods:
            try:
                text = method()
                if text:
                    # Count meaningful words
                    words = len([w for w in text.split() if len(w) > 2 and w.isalpha()])
                    if words > best_word_count:
                        best_text = text
                        best_word_count = words
            except:
                continue
        
        return best_text
    
    def _extract_from_dict(self, page) -> str:
        """Extract using text dictionary method"""
        text_dict = page.get_text("dict")
        lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    if "spans" in line:
                        for span in line["spans"]:
                            line_text += span.get("text", "")
                    if line_text.strip():
                        lines.append(line_text.strip())
        
        return "\n".join(lines)
    
    def _extract_from_blocks(self, page) -> str:
        """Extract using blocks method"""
        blocks = page.get_text("blocks")
        text_parts = []
        
        for block in blocks:
            if len(block) > 4 and isinstance(block[4], str):
                text_parts.append(block[4].strip())
        
        return "\n".join(text_parts)
    
    def _clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are just page numbers
            if re.match(r'^\d+$', line):
                continue
            
            # Skip lines that are mostly dots or 'o' characters (table of contents)
            if re.match(r'^[o\.\s]+$', line):
                continue
            
            # Skip very short lines unless they're meaningful
            if len(line) < 3:
                continue
            
            # Keep lines that have meaningful content
            meaningful_chars = len(re.findall(r'[a-zA-Z0-9]', line))
            if meaningful_chars >= 3:  # At least 3 alphanumeric characters
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_token_based_chunks_with_overlap(self, text: str) -> List[Dict]:
        """Create overlapping chunks based on token count"""
        print(f"Creating token-based chunks (max {self.max_tokens_per_chunk} tokens, {self.overlap_tokens} overlap)...")
        
        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max tokens, save current chunk
            if current_chunk_tokens + sentence_tokens > self.max_tokens_per_chunk and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "tokens": current_chunk_tokens,
                    "sentence_start": i - len(current_chunk_sentences),
                    "sentence_end": i - 1
                })
                
                # Create overlap for next chunk
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences, self.overlap_tokens)
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = sum(self.count_tokens(s) for s in overlap_sentences)
                
            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            i += 1
        
        # Add the last chunk if it has content
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "tokens": current_chunk_tokens,
                "sentence_start": len(sentences) - len(current_chunk_sentences),
                "sentence_end": len(sentences) - 1
            })
        
        print(f"Created {len(chunks)} overlapping chunks")
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.update({
                "chunk_id": i,
                "word_count": len(chunk["text"].split()),
                "char_count": len(chunk["text"])
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        # Split by sentence endings, but be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep substantial sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_overlap_sentences(self, sentences: List[str], target_overlap_tokens: int) -> List[str]:
        """Get the last few sentences for overlap, staying within token limit"""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings in batches for better performance"""
        print(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} items)...")
            
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting between batches
                if batch_num < total_batches:
                    time.sleep(0.5)  # Small delay between batches
                    
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                # Add empty embeddings for failed batch to maintain alignment
                all_embeddings.extend([None] * len(batch_texts))
        
        print(f"Generated {len([e for e in all_embeddings if e is not None])} embeddings successfully")
        return all_embeddings
    
    def process_pdf_advanced(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Advanced PDF processing pipeline with token-based chunking and batch embeddings"""
        # Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Extract all meaningful content
        combined_content, page_markers = self.extract_content_comprehensive(pdf_path)
        
        if not combined_content.strip():
            print("No meaningful content extracted!")
            return []
        
        # Create token-based chunks with overlap
        chunks = self.create_token_based_chunks_with_overlap(combined_content)
        
        if not chunks:
            print("No chunks created!")
            return []
        
        # Prepare texts for batch embedding
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(chunk_texts, batch_size=50)  # Smaller batches for reliability
        
        # Create final chunks with embeddings
        final_chunks = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                print(f"Skipping chunk {i} due to embedding failure")
                continue
            
            # Determine which pages this chunk spans
            chunk_pages = self._determine_chunk_pages(chunk["text"], page_markers)
            
            # Extract meaningful section titles from chunk content
            section_titles = self._extract_section_titles(chunk["text"])
            section_title = section_titles[0] if section_titles else f"{manual_id} - Chunk {i+1}"
            
            # Clean up section title if too long
            if len(section_title) > 80:
                section_title = section_title[:77] + "..."
            
            final_chunk = {
                "manual_id": manual_id,
                "section_title": section_title,
                "chunk_text": chunk["text"],
                "chunk_index": i,
                "total_chunks_in_section": len(chunks),
                "page_start": min(chunk_pages) if chunk_pages else 1,
                "page_end": max(chunk_pages) if chunk_pages else 1,
                "word_count": chunk["word_count"],
                "token_count": chunk["tokens"],
                "char_count": chunk["char_count"],
                "embedding_vector": embedding,
                "embedding_dimensions": len(embedding),
                "has_overlap": i > 0,  # All chunks except first have overlap
                "overlap_tokens": self.overlap_tokens if i > 0 else 0,
                "extracted_sections": section_titles  # Keep all found sections for reference
            }
            
            final_chunks.append(final_chunk)
        
        return final_chunks
    
    def _determine_chunk_pages(self, chunk_text: str, page_markers: Dict) -> List[int]:
        """Determine which pages a chunk spans based on page markers"""
        pages = []
        for page_num, (start_pos, end_pos) in page_markers.items():
            if f"[Page {page_num}]" in chunk_text:
                pages.append(page_num)
        return pages if pages else [1]
    
    def _extract_section_titles(self, text: str) -> List[str]:
        """Extract section titles from text for better chunk labeling"""
        lines = text.split('\n')
        section_titles = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered sections, capitalized headers, etc.
            if re.match(r'^\d+\.?\d*\s+[A-Z][A-Za-z\s]+', line):
                section_titles.append(line)
            elif line.isupper() and len(line) > 5 and len(line) < 50:
                section_titles.append(line)
            elif re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*\s*$', line) and len(line) > 10:
                section_titles.append(line)
        
        return section_titles

def main():
    # Initialize advanced processor
    processor = AdvancedPDFProcessor(
        max_tokens_per_chunk=400,  # Optimal size for embeddings
        overlap_tokens=50          # 50 token overlap between chunks
    )
    
    # PDF configurations
    pdfs = [
        {
            "url": "https://qyihtrkdkcsxzprvbtyy.supabase.co/storage/v1/object/public/files//QD17010_Radical_SR3_XXR_Owners_Manual.pdf",
            "filename": "Radical_SR3_Owners_Manual.pdf",
            "manual_id": "Radical_SR3_Owners_Manual"
        },
        {
            "url": "https://qyihtrkdkcsxzprvbtyy.supabase.co/storage/v1/object/public/files//Radical_Handling_Guide_Master_v1.1.pdf",
            "filename": "Radical_Handling_Guide.pdf",
            "manual_id": "Radical_Handling_Guide"
        }
    ]
    
    all_chunks = []
    
    for pdf_config in pdfs:
        print(f"\n{'='*70}")
        print(f"Processing {pdf_config['manual_id']}")
        print(f"{'='*70}")
        
        chunks = processor.process_pdf_advanced(
            pdf_config["url"],
            pdf_config["filename"],
            pdf_config["manual_id"]
        )
        
        print(f"Generated {len(chunks)} advanced chunks")
        
        # Show statistics
        if chunks:
            word_counts = [c['word_count'] for c in chunks]
            token_counts = [c['token_count'] for c in chunks]
            
            print(f"Average words per chunk: {sum(word_counts)/len(word_counts):.1f}")
            print(f"Average tokens per chunk: {sum(token_counts)/len(token_counts):.1f}")
            print(f"Token range: {min(token_counts)}-{max(token_counts)}")
            print(f"Chunks with overlap: {len([c for c in chunks if c['has_overlap']])}")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  {i+1}. Pages {chunk['page_start']}-{chunk['page_end']}")
            print(f"     {chunk['word_count']} words, {chunk['token_count']} tokens")
            print(f"     Overlap: {chunk['overlap_tokens']} tokens")
            print(f"     Preview: {chunk['chunk_text'][:120]}...")
            print()
        
        all_chunks.extend(chunks)
    
    print(f"\n{'='*70}")
    print(f"FINAL ADVANCED RESULTS")
    print(f"{'='*70}")
    print(f"Total advanced chunks: {len(all_chunks)}")
    
    if all_chunks:
        # Overall statistics
        all_word_counts = [c['word_count'] for c in all_chunks]
        all_token_counts = [c['token_count'] for c in all_chunks]
        
        print(f"Overall average words per chunk: {sum(all_word_counts)/len(all_word_counts):.1f}")
        print(f"Overall average tokens per chunk: {sum(all_token_counts)/len(all_token_counts):.1f}")
        print(f"Total chunks with overlap: {len([c for c in all_chunks if c['has_overlap']])}")
        print(f"Total tokens processed: {sum(all_token_counts)}")
    
    # Save advanced chunks
    output_file = "chunks_with_embeddings_advanced.json"
    with open(output_file, "w") as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Advanced chunks with embeddings saved to {output_file}")
    
    # Search for brake-related content
    brake_chunks = [c for c in all_chunks if 'brake' in c['chunk_text'].lower()]
    print(f"\nBrake-related chunks found: {len(brake_chunks)}")
    for chunk in brake_chunks[:3]:
        print(f"- Tokens: {chunk['token_count']}, Overlap: {chunk['overlap_tokens']}")
        print(f"  Content: {chunk['chunk_text'][:100]}...")

if __name__ == "__main__":
    main()