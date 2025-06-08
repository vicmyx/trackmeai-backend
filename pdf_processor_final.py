import os
import requests
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict
import re
import json

class FinalPDFProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
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
    
    def extract_content_comprehensive(self, pdf_path: str) -> List[Dict]:
        """Extract all meaningful content from PDF"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        print(f"Processing {doc.page_count} pages...")
        
        # Process every page and look for substantial content
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
            
            # Create chunks for this page if it has good content
            page_chunks = self._create_page_chunks(cleaned_text, page_num + 1)
            chunks.extend(page_chunks)
        
        doc.close()
        
        print(f"Extracted {len(chunks)} content chunks")
        return chunks
    
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
    
    def _create_page_chunks(self, text: str, page_num: int) -> List[Dict]:
        """Create meaningful chunks from page text"""
        chunks = []
        
        # Split by sections if possible
        sections = self._identify_sections(text)
        
        if sections:
            for section_title, section_content in sections:
                if len(section_content.strip()) > 30:
                    chunk = {
                        "content": section_content.strip(),
                        "title": section_title,
                        "page": page_num,
                        "type": "section"
                    }
                    chunks.append(chunk)
        else:
            # If no sections found, create chunks by content length
            if len(text.strip()) > 50:
                chunk = {
                    "content": text.strip(),
                    "title": f"Page {page_num} Content",
                    "page": page_num,
                    "type": "page"
                }
                chunks.append(chunk)
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """Identify sections within text"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            is_header = (
                # Numbered sections
                re.match(r'^\d+\.?\s+[A-Z]', line) or
                # All caps headers
                (line.isupper() and len(line) > 3 and len(line) < 80) or
                # Specific patterns
                re.match(r'^\d+\.\d+\s+', line) or
                re.match(r'^[A-Z][A-Z\s&/]+[A-Z]$', line)
            )
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content) > 20:
                        sections.append((current_section, content))
                
                # Start new section
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if len(content) > 20:
                sections.append((current_section, content))
        
        return sections
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def process_pdf_final(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Final PDF processing pipeline"""
        # Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Extract all meaningful content
        raw_chunks = self.extract_content_comprehensive(pdf_path)
        
        # Convert to final format
        final_chunks = []
        
        for i, chunk in enumerate(raw_chunks):
            # Skip chunks that are too short
            if len(chunk["content"]) < 50:
                continue
            
            # Create final chunk format
            final_chunk = {
                "manual_id": manual_id,
                "section_title": chunk["title"],
                "chunk_text": chunk["content"],
                "chunk_index": i,
                "page_start": chunk["page"],
                "word_count": len(chunk["content"].split()),
                "token_count": self.count_tokens(chunk["content"])
            }
            
            final_chunks.append(final_chunk)
        
        return final_chunks

def main():
    processor = FinalPDFProcessor()
    
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
        print(f"\n{'='*60}")
        print(f"Processing {pdf_config['manual_id']}")
        print(f"{'='*60}")
        
        chunks = processor.process_pdf_final(
            pdf_config["url"],
            pdf_config["filename"],
            pdf_config["manual_id"]
        )
        
        print(f"Generated {len(chunks)} high-quality chunks")
        
        # Show statistics
        if chunks:
            word_counts = [c['word_count'] for c in chunks]
            token_counts = [c['token_count'] for c in chunks]
            
            print(f"Average words per chunk: {sum(word_counts)/len(word_counts):.1f}")
            print(f"Average tokens per chunk: {sum(token_counts)/len(token_counts):.1f}")
            print(f"Chunks with >50 words: {len([c for c in chunks if c['word_count'] > 50])}")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  {i+1}. {chunk['section_title']} (Page {chunk['page_start']})")
            print(f"     {chunk['word_count']} words, {chunk['token_count']} tokens")
            print(f"     Preview: {chunk['chunk_text'][:120]}...")
            print()
        
        all_chunks.extend(chunks)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total high-quality chunks: {len(all_chunks)}")
    
    if all_chunks:
        # Overall statistics
        all_word_counts = [c['word_count'] for c in all_chunks]
        print(f"Overall average words per chunk: {sum(all_word_counts)/len(all_word_counts):.1f}")
        print(f"Substantial chunks (>50 words): {len([c for c in all_chunks if c['word_count'] > 50])}")
        print(f"Detailed chunks (>100 words): {len([c for c in all_chunks if c['word_count'] > 100])}")
    
    # Save final chunks
    with open("processed_chunks_final.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Final chunks saved to processed_chunks_final.json")
    
    # Search for brake-related content
    brake_chunks = [c for c in all_chunks if 'brake' in c['chunk_text'].lower()]
    print(f"\nBrake-related chunks found: {len(brake_chunks)}")
    for chunk in brake_chunks[:3]:
        print(f"- {chunk['section_title']}: {chunk['chunk_text'][:100]}...")

if __name__ == "__main__":
    main()