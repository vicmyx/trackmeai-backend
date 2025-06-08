import os
import requests
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict, Tuple
import re
import json

class EnhancedPDFProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
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
    
    def extract_text_advanced(self, pdf_path: str) -> List[Dict]:
        """Advanced text extraction using multiple methods"""
        doc = fitz.open(pdf_path)
        all_content = []
        
        print(f"Processing {doc.page_count} pages...")
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_content = self._extract_page_content(page, page_num + 1)
            
            if page_content and len(page_content.strip()) > 20:
                all_content.append({
                    "page_number": page_num + 1,
                    "content": page_content,
                    "word_count": len(page_content.split())
                })
        
        doc.close()
        
        # Group content into logical sections
        sections = self._group_into_sections(all_content)
        
        return sections
    
    def _extract_page_content(self, page, page_num: int) -> str:
        """Extract content from a single page using multiple methods"""
        methods = [
            ("standard", lambda: page.get_text()),
            ("blocks", lambda: self._extract_text_blocks(page)),
            ("dict", lambda: self._extract_text_dict(page)),
            ("html", lambda: self._extract_text_html(page))
        ]
        
        best_content = ""
        best_word_count = 0
        
        for method_name, method_func in methods:
            try:
                content = method_func()
                content = self._clean_text(content)
                word_count = len([w for w in content.split() if len(w) > 2 and w.isalpha()])
                
                if word_count > best_word_count:
                    best_content = content
                    best_word_count = word_count
                    
            except Exception as e:
                print(f"Method {method_name} failed for page {page_num}: {e}")
                continue
        
        return best_content
    
    def _extract_text_blocks(self, page) -> str:
        """Extract text using blocks method"""
        blocks = page.get_text("blocks")
        text_parts = []
        
        for block in blocks:
            if len(block) > 4 and isinstance(block[4], str):
                text_parts.append(block[4])
        
        return " ".join(text_parts)
    
    def _extract_text_dict(self, page) -> str:
        """Extract text using dictionary method"""
        text_dict = page.get_text("dict")
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if "text" in span:
                                text_parts.append(span["text"])
        
        return " ".join(text_parts)
    
    def _extract_text_html(self, page) -> str:
        """Extract text using HTML method and parse"""
        import re
        html = page.get_text("html")
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[•○◦▪▫■□▲△▼▽◆◇★☆●○]+', ' ', text)  # Remove bullet points
        text = re.sub(r'[\.]{3,}', ' ', text)  # Remove multiple dots
        text = re.sub(r'\s+o\s+', ' ', text)  # Remove isolated 'o' characters
        
        # Clean up page numbers and headers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip very short lines
            if len(line) < 3:
                continue
                
            # Skip lines that are just numbers
            if re.match(r'^\d+$', line):
                continue
                
            # Skip lines with only special characters
            if re.match(r'^[^\w\s]+$', line):
                continue
            
            # Skip lines that are mostly dots or symbols
            if len(re.sub(r'[^a-zA-Z0-9\s]', '', line)) < len(line) * 0.3:
                continue
                
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _group_into_sections(self, all_content: List[Dict]) -> List[Dict]:
        """Group pages into logical sections"""
        if not all_content:
            return []
        
        sections = []
        current_section = {
            "title": f"Section 1 (Page {all_content[0]['page_number']})",
            "content": "",
            "page_start": all_content[0]['page_number'],
            "page_end": all_content[0]['page_number'],
            "word_count": 0
        }
        
        pages_per_section = max(2, len(all_content) // 10)  # Aim for ~10 sections
        
        for i, page_data in enumerate(all_content):
            # Add content to current section
            current_section["content"] += page_data["content"] + " "
            current_section["page_end"] = page_data["page_number"]
            current_section["word_count"] += page_data["word_count"]
            
            # Start new section when we have enough content or pages
            should_split = (
                (i + 1) % pages_per_section == 0 or  # Regular interval
                current_section["word_count"] > 1000 or  # Enough content
                i == len(all_content) - 1  # Last page
            )
            
            if should_split and current_section["content"].strip():
                # Clean up the section content
                current_section["content"] = current_section["content"].strip()
                
                # Update title to reflect page range
                if current_section["page_start"] != current_section["page_end"]:
                    current_section["title"] = f"Section {len(sections) + 1} (Pages {current_section['page_start']}-{current_section['page_end']})"
                
                sections.append(current_section)
                
                # Start new section if not at the end
                if i < len(all_content) - 1:
                    next_page = all_content[i + 1]["page_number"]
                    current_section = {
                        "title": f"Section {len(sections) + 1} (Page {next_page})",
                        "content": "",
                        "page_start": next_page,
                        "page_end": next_page,
                        "word_count": 0
                    }
        
        return sections
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_text_semantic(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into semantic chunks"""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Try to split by paragraphs first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Fall back to sentence splitting
        return self._chunk_by_sentences(text, max_tokens)
    
    def _chunk_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Split text by sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdf_enhanced(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Enhanced PDF processing pipeline"""
        # Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Extract text with advanced methods
        sections = self.extract_text_advanced(pdf_path)
        
        print(f"Extracted {len(sections)} sections from {manual_id}")
        
        # Process each section into chunks
        all_chunks = []
        
        for section in sections:
            if section["word_count"] < 10:  # Skip sections with too little content
                continue
                
            section_chunks = self.chunk_text_semantic(section["content"])
            
            for i, chunk in enumerate(section_chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                chunk_data = {
                    "manual_id": manual_id,
                    "section_title": section["title"],
                    "chunk_text": chunk.strip(),
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "page_start": section["page_start"],
                    "page_end": section.get("page_end", section["page_start"]),
                    "word_count": len(chunk.split())
                }
                all_chunks.append(chunk_data)
        
        return all_chunks

def main():
    processor = EnhancedPDFProcessor()
    
    # PDF URLs and configurations
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
    
    all_processed_chunks = []
    
    for pdf_config in pdfs:
        print(f"\n{'='*50}")
        print(f"Processing {pdf_config['manual_id']}...")
        print(f"{'='*50}")
        
        chunks = processor.process_pdf_enhanced(
            pdf_config["url"],
            pdf_config["filename"],
            pdf_config["manual_id"]
        )
        
        print(f"Generated {len(chunks)} chunks for {pdf_config['manual_id']}")
        
        # Show sample chunks
        print(f"\nSample chunks from {pdf_config['manual_id']}:")
        for i, chunk in enumerate(chunks[:3]):
            token_count = processor.count_tokens(chunk["chunk_text"])
            print(f"  Chunk {i+1}: {token_count} tokens, {chunk['word_count']} words")
            print(f"  Section: {chunk['section_title']}")
            print(f"  Content: {chunk['chunk_text'][:150]}...")
            print()
        
        all_processed_chunks.extend(chunks)
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total chunks generated: {len(all_processed_chunks)}")
    
    # Analyze chunk quality
    word_counts = [chunk['word_count'] for chunk in all_processed_chunks]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Chunks with >20 words: {len([c for c in all_processed_chunks if c['word_count'] > 20])}")
    print(f"Chunks with >50 words: {len([c for c in all_processed_chunks if c['word_count'] > 50])}")
    
    # Save enhanced chunks
    with open("processed_chunks_enhanced.json", "w") as f:
        json.dump(all_processed_chunks, f, indent=2)
    
    print(f"Enhanced chunks saved to processed_chunks_enhanced.json")
    
    # Show some brake-related content if found
    brake_chunks = [c for c in all_processed_chunks if 'brake' in c['chunk_text'].lower()]
    if brake_chunks:
        print(f"\nFound {len(brake_chunks)} brake-related chunks:")
        for chunk in brake_chunks[:2]:
            print(f"- {chunk['section_title']}: {chunk['chunk_text'][:100]}...")

if __name__ == "__main__":
    main()