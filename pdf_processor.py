import os
import requests
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict, Tuple
import re

class PDFProcessor:
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
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with multiple extraction methods"""
        doc = fitz.open(pdf_path)
        sections = []
        current_section = {"title": "Introduction", "text": "", "page_start": 1}
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Try multiple text extraction methods
            text = page.get_text()
            
            # If standard extraction fails or gives mostly symbols, try textpage
            if self._is_mostly_symbols(text):
                try:
                    text = page.get_text("text")
                except:
                    # Fallback to OCR-like extraction
                    try:
                        textpage = page.get_textpage()
                        text = textpage.extractText()
                    except:
                        # Last resort: get text with layout
                        text = page.get_text("blocks")
                        if isinstance(text, list):
                            text = " ".join([block[4] for block in text if len(block) > 4])
            
            # Clean and filter the text
            text = self._clean_text(text)
            
            if not text or len(text.strip()) < 10:
                continue
                
            # Basic section detection - look for lines that might be headings
            lines = text.split('\n')
            page_text = ""
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Skip lines that are mostly symbols or numbers
                if self._is_mostly_symbols(line) or re.match(r'^[\d\.\s\-_]+$', line):
                    continue
                
                # Detect potential section headers
                if (len(line) < 100 and len(line) > 5 and
                    (line.isupper() or 
                     re.match(r'^\d+\.?\d*\s+[A-Z]', line) or
                     re.match(r'^Chapter\s+\d+', line, re.IGNORECASE) or
                     re.match(r'^Section\s+\d+', line, re.IGNORECASE) or
                     re.match(r'^[A-Z][A-Z\s]+[A-Z]$', line))):
                    
                    # Save previous section if it has substantial content
                    if len(current_section["text"].strip()) > 50:
                        sections.append(current_section.copy())
                    
                    # Start new section
                    current_section = {
                        "title": line,
                        "text": "",
                        "page_start": page_num + 1
                    }
                else:
                    page_text += line + " "
            
            # Add page text to current section
            if page_text.strip():
                current_section["text"] += page_text + " "
        
        # Add the last section if it has content
        if len(current_section["text"].strip()) > 50:
            sections.append(current_section)
        
        doc.close()
        
        # If no substantial sections were detected, create page-based sections
        if len(sections) <= 1 or all(len(s["text"].strip()) < 100 for s in sections):
            sections = self._create_page_based_sections(pdf_path)
        
        return sections
    
    def _is_mostly_symbols(self, text: str) -> bool:
        """Check if text is mostly symbols or bullet points"""
        if not text or len(text.strip()) < 5:
            return True
        
        # Count actual words vs symbols
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        symbols = re.findall(r'[•○◦▪▫■□▲△▼▽◆◇★☆●○]', text)
        
        # If we have very few actual words compared to symbols
        return len(words) < 3 and len(symbols) > len(words)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely page numbers or artifacts
            if len(line) < 3:
                continue
            # Skip lines that are just numbers
            if re.match(r'^\d+$', line):
                continue
            # Skip lines with mostly special characters
            if len(re.sub(r'[^a-zA-Z0-9\s]', '', line)) < len(line) * 0.5:
                continue
                
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _create_page_based_sections(self, pdf_path: str) -> List[Dict]:
        """Create page-based sections when no clear structure is detected"""
        doc = fitz.open(pdf_path)
        sections = []
        pages_per_section = max(3, doc.page_count // 15)  # Aim for smaller sections
        
        for i in range(0, doc.page_count, pages_per_section):
            end_page = min(i + pages_per_section, doc.page_count)
            combined_text = ""
            
            for page_num in range(i, end_page):
                page = doc[page_num]
                
                # Try different extraction methods for each page
                page_text = page.get_text()
                
                # Try blocks extraction if regular text is poor
                if self._is_mostly_symbols(page_text):
                    try:
                        blocks = page.get_text("blocks")
                        if isinstance(blocks, list):
                            page_text = " ".join([block[4] for block in blocks if len(block) > 4 and isinstance(block[4], str)])
                    except:
                        pass
                
                # Clean the page text
                page_text = self._clean_text(page_text)
                if page_text and len(page_text.strip()) > 10:
                    combined_text += page_text + " "
            
            if combined_text.strip() and len(combined_text.strip()) > 20:
                sections.append({
                    "title": f"Section {len(sections) + 1} (Pages {i+1}-{end_page})",
                    "text": combined_text.strip(),
                    "page_start": i + 1
                })
        
        doc.close()
        return sections
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> List[str]:
        """Split text into chunks with token-based splitting"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                if self.count_tokens(sentence) <= max_tokens:
                    current_chunk = sentence
                else:
                    # If single sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        test_word_chunk = temp_chunk + " " + word if temp_chunk else word
                        if self.count_tokens(test_word_chunk) <= max_tokens:
                            temp_chunk = test_word_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    current_chunk = temp_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdf(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Complete PDF processing pipeline"""
        # Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Extract text with sections
        sections = self.extract_text_from_pdf(pdf_path)
        
        # Process each section into chunks
        all_chunks = []
        
        for section in sections:
            section_chunks = self.chunk_text(section["text"])
            
            for i, chunk in enumerate(section_chunks):
                chunk_data = {
                    "manual_id": manual_id,
                    "section_title": section["title"],
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "page_start": section["page_start"]
                }
                all_chunks.append(chunk_data)
        
        return all_chunks

def main():
    processor = PDFProcessor()
    
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
        print(f"\nProcessing {pdf_config['manual_id']}...")
        chunks = processor.process_pdf(
            pdf_config["url"],
            pdf_config["filename"],
            pdf_config["manual_id"]
        )
        
        print(f"Generated {len(chunks)} chunks for {pdf_config['manual_id']}")
        
        # Show sample chunks
        print(f"\nSample chunks from {pdf_config['manual_id']}:")
        for i, chunk in enumerate(chunks[:3]):
            token_count = processor.count_tokens(chunk["chunk_text"])
            print(f"  Chunk {i+1}: {token_count} tokens")
            print(f"  Section: {chunk['section_title']}")
            print(f"  Text preview: {chunk['chunk_text'][:100]}...")
            print()
        
        all_processed_chunks.extend(chunks)
    
    print(f"\nTotal chunks generated: {len(all_processed_chunks)}")
    
    # Save chunks to JSON for inspection
    import json
    with open("processed_chunks.json", "w") as f:
        json.dump(all_processed_chunks, f, indent=2)
    
    print("Chunks saved to processed_chunks.json")

if __name__ == "__main__":
    main()