import os
import requests
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict, Tuple, Optional
import re
import json
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class StructuredPDFProcessor:
    """Enhanced PDF processor following racing manual best practices"""
    
    def __init__(self, max_tokens_per_chunk: int = 400, overlap_tokens: int = 50):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        
        # Advanced regex patterns for structure detection
        self.heading_patterns = [
            re.compile(r'^\d+\.\d+\s+.*$', re.MULTILINE),  # 7.17 STANDARD PRE-RACE
            re.compile(r'^\d+\s+[A-Z][A-Z\s]{2,}$', re.MULTILINE),  # 7 MAINTENANCE
            re.compile(r'^[A-Z][A-Z\s]{5,}$', re.MULTILINE),  # ALL CAPS HEADINGS
        ]
        
        self.bullet_patterns = [
            re.compile(r'^[-*•]\s+(.+)$', re.MULTILINE),  # Bullet points
            re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE),  # Numbered lists
        ]
        
        self.checklist_patterns = [
            re.compile(r'CHECK.*:', re.IGNORECASE),
            re.compile(r'INSPECT.*:', re.IGNORECASE),
            re.compile(r'VERIFY.*:', re.IGNORECASE),
        ]
        
        # Initialize OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        self.openai_client = openai.OpenAI(api_key=api_key)
    
    def download_pdf(self, url: str, filename: str) -> str:
        """Download PDF from URL and save locally"""
        print(f"📥 Downloading {filename}...")
        
        response = requests.get(url)
        response.raise_for_status()
        
        os.makedirs("pdfs", exist_ok=True)
        filepath = f"pdfs/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {filepath}")
        return filepath
    
    def extract_table_of_contents(self, pdf_path: str) -> Dict[str, Dict]:
        """Extract and parse table of contents to build section index"""
        doc = fitz.open(pdf_path)
        toc_data = {}
        
        print(f"🔍 Parsing table of contents...")
        
        # Look for TOC in first 10 pages
        for page_num in range(min(10, doc.page_count)):
            page = doc[page_num]
            text = page.get_text()
            
            # Look for TOC indicators
            if any(indicator in text.lower() for indicator in ['contents', 'table of contents', 'index']):
                print(f"📋 Found TOC on page {page_num + 1}")
                toc_entries = self._parse_toc_page(text, page_num + 1)
                toc_data.update(toc_entries)
        
        # Also extract from PyMuPDF's built-in TOC
        try:
            outline = doc.get_toc()
            if outline:
                print(f"📖 Found {len(outline)} outline entries")
                for level, title, page_num in outline:
                    clean_title = title.strip()
                    if clean_title and page_num > 0:
                        toc_data[clean_title] = {
                            "title": clean_title,
                            "page": page_num,
                            "level": level
                        }
        except:
            pass
        
        doc.close()
        print(f"✅ Extracted {len(toc_data)} TOC entries")
        return toc_data
    
    def _parse_toc_page(self, text: str, page_num: int) -> Dict[str, Dict]:
        """Parse TOC page text to extract section titles and page numbers"""
        toc_entries = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Pattern: "Section Title ... Page Number"
            # Example: "7.17 STANDARD PRE-RACE/TEST CHECKLIST ........... 89"
            match = re.search(r'^(.+?)\s*[.\s]*(\d+)$', line)
            if match:
                title = match.group(1).strip()
                target_page = int(match.group(2))
                
                # Filter out non-section entries
                if len(title) > 5 and not title.lower().startswith('page'):
                    # Clean up title (remove dots, extra spaces)
                    clean_title = re.sub(r'[.\s]+$', '', title).strip()
                    
                    toc_entries[clean_title] = {
                        "title": clean_title,
                        "page": target_page,
                        "level": self._determine_section_level(clean_title)
                    }
        
        return toc_entries
    
    def _determine_section_level(self, title: str) -> int:
        """Determine hierarchical level of section title"""
        if re.match(r'^\d+\.\d+\.\d+', title):  # 7.17.1
            return 3
        elif re.match(r'^\d+\.\d+', title):     # 7.17
            return 2
        elif re.match(r'^\d+\s', title):        # 7 MAINTENANCE
            return 1
        else:
            return 0
    
    def extract_structured_content(self, pdf_path: str, toc_data: Dict) -> List[Dict]:
        """Extract content with structure awareness using TOC"""
        doc = fitz.open(pdf_path)
        structured_content = []
        
        print(f"📚 Extracting structured content from {doc.page_count} pages...")
        
        # Sort TOC entries by page number
        sorted_sections = sorted(toc_data.items(), key=lambda x: x[1]['page'])
        
        for i, (section_title, section_info) in enumerate(sorted_sections):
            start_page = section_info['page'] - 1  # Convert to 0-indexed
            
            # Determine end page (next section's start or document end)
            if i + 1 < len(sorted_sections):
                end_page = sorted_sections[i + 1][1]['page'] - 1
            else:
                end_page = doc.page_count
            
            # Extract content for this section
            section_content = self._extract_section_content(
                doc, section_title, start_page, end_page, section_info['level']
            )
            
            if section_content:
                structured_content.extend(section_content)
        
        # Also extract pages not covered by TOC
        covered_pages = set()
        for content in structured_content:
            covered_pages.update(range(content['page_start'], content['page_end'] + 1))
        
        uncovered_pages = set(range(doc.page_count)) - covered_pages
        if uncovered_pages:
            print(f"📄 Processing {len(uncovered_pages)} uncovered pages...")
            for page_num in sorted(uncovered_pages):
                orphan_content = self._extract_orphan_page_content(doc, page_num)
                if orphan_content:
                    structured_content.extend(orphan_content)
        
        doc.close()
        print(f"✅ Extracted {len(structured_content)} structured content blocks")
        return structured_content
    
    def _extract_section_content(self, doc, section_title: str, start_page: int, end_page: int, level: int) -> List[Dict]:
        """Extract content for a specific section"""
        content_blocks = []
        section_text = ""
        
        # Extract text from all pages in section
        for page_num in range(start_page, min(end_page, doc.page_count)):
            if page_num < 0:
                continue
                
            page = doc[page_num]
            page_text = self._extract_best_text(page)
            
            if page_text and len(page_text.strip()) > 30:
                cleaned_text = self._clean_text_advanced(page_text)
                if cleaned_text:
                    section_text += f"\n[Page {page_num + 1}]\n{cleaned_text}"
        
        if not section_text.strip():
            return []
        
        # Detect subsections and checklists within the section
        subsections = self._parse_subsections(section_text, section_title)
        
        # Create chunks for this section
        if subsections:
            # Process each subsection separately
            for i, subsection in enumerate(subsections):
                chunks = self._create_section_chunks(
                    subsection['content'], 
                    f"{section_title} - {subsection['title']}", 
                    start_page + 1, 
                    min(end_page, doc.page_count),
                    level + 1,
                    subsection
                )
                content_blocks.extend(chunks)
        else:
            # Process entire section as one unit
            chunks = self._create_section_chunks(
                section_text, 
                section_title, 
                start_page + 1, 
                min(end_page, doc.page_count),
                level
            )
            content_blocks.extend(chunks)
        
        return content_blocks
    
    def _parse_subsections(self, text: str, parent_title: str) -> List[Dict]:
        """Parse subsections within a section"""
        subsections = []
        lines = text.split('\n')
        current_subsection = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a subsection heading
            is_heading = False
            for pattern in self.heading_patterns:
                if pattern.match(line):
                    is_heading = True
                    break
            
            # Also check for numbered items that might be subsections
            if re.match(r'^\d+\.\s+[A-Z]', line):
                is_heading = True
            
            if is_heading:
                # Save previous subsection
                if current_subsection and current_content:
                    current_subsection['content'] = '\n'.join(current_content)
                    subsections.append(current_subsection)
                
                # Start new subsection
                current_subsection = {
                    'title': line,
                    'content': '',
                    'type': self._classify_subsection_type(line)
                }
                current_content = []
            else:
                if current_subsection:
                    current_content.append(line)
        
        # Save last subsection
        if current_subsection and current_content:
            current_subsection['content'] = '\n'.join(current_content)
            subsections.append(current_subsection)
        
        return subsections
    
    def _classify_subsection_type(self, title: str) -> str:
        """Classify the type of subsection"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['checklist', 'check', 'inspect']):
            return 'checklist'
        elif any(word in title_lower for word in ['procedure', 'step', 'how to']):
            return 'procedure'
        elif any(word in title_lower for word in ['spec', 'specification', 'parameter']):
            return 'specification'
        elif any(word in title_lower for word in ['setup', 'adjustment', 'config']):
            return 'setup'
        else:
            return 'general'
    
    def _create_section_chunks(self, content: str, section_title: str, start_page: int, end_page: int, level: int, subsection_info: Optional[Dict] = None) -> List[Dict]:
        """Create chunks for a section with proper metadata"""
        if not content.strip():
            return []
        
        # Extract structured data based on section type
        structured_data = {}
        if subsection_info and subsection_info['type'] == 'checklist':
            structured_data = self._extract_checklist_items(content)
        elif subsection_info and subsection_info['type'] == 'procedure':
            structured_data = self._extract_procedure_steps(content)
        elif subsection_info and subsection_info['type'] == 'specification':
            structured_data = self._extract_specifications(content)
        
        # Create token-based chunks
        chunks = self._create_token_chunks(content)
        
        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "section_title": section_title,
                "chunk_text": chunk['text'],
                "chunk_index": i,
                "total_chunks_in_section": len(chunks),
                "page_start": start_page,
                "page_end": end_page,
                "section_level": level,
                "word_count": len(chunk['text'].split()),
                "token_count": self.count_tokens(chunk['text']),
                "char_count": len(chunk['text']),
                "section_type": subsection_info['type'] if subsection_info else 'general',
                "structured_data": structured_data,
                "has_overlap": i > 0,
                "overlap_tokens": self.overlap_tokens if i > 0 else 0
            }
            final_chunks.append(chunk_data)
        
        return final_chunks
    
    def _extract_checklist_items(self, content: str) -> Dict:
        """Extract checklist items as structured data"""
        items = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match various checklist patterns
            for pattern in self.bullet_patterns:
                match = pattern.match(line)
                if match:
                    item_text = match.group(1).strip()
                    if len(item_text) > 3:
                        items.append({
                            "text": item_text,
                            "type": "check_item"
                        })
                    break
            
            # Also look for "Check..." statements
            if any(pattern.search(line) for pattern in self.checklist_patterns):
                items.append({
                    "text": line,
                    "type": "check_instruction"
                })
        
        return {"checklist_items": items, "total_items": len(items)}
    
    def _extract_procedure_steps(self, content: str) -> Dict:
        """Extract procedure steps as structured data"""
        steps = []
        lines = content.split('\n')
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered steps
            step_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if step_match:
                if current_step:
                    steps.append(current_step)
                
                current_step = {
                    "step_number": int(step_match.group(1)),
                    "instruction": step_match.group(2).strip(),
                    "details": []
                }
            elif current_step and line:
                # Add details to current step
                current_step["details"].append(line)
        
        if current_step:
            steps.append(current_step)
        
        return {"procedure_steps": steps, "total_steps": len(steps)}
    
    def _extract_specifications(self, content: str) -> Dict:
        """Extract specifications as structured data"""
        specs = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for key-value patterns
            # Pattern: "Pressure: 28 psi"
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                specs[key] = value
        
        return {"specifications": specs, "total_specs": len(specs)}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def _create_token_chunks(self, text: str) -> List[Dict]:
        """Create overlapping chunks based on token count"""
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
                    "tokens": current_chunk_tokens
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
                "tokens": current_chunk_tokens
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_overlap_sentences(self, sentences: List[str], target_overlap_tokens: int) -> List[str]:
        """Get the last few sentences for overlap, staying within token limit"""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
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
        """Advanced text cleaning with structure preservation"""
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
            
            # Skip lines that are mostly dots (table of contents)
            if re.match(r'^[o\.\s]+$', line):
                continue
            
            # Skip very short lines unless they're meaningful headings
            if len(line) < 3 and not re.match(r'^\d+\.', line):
                continue
            
            # Keep lines that have meaningful content
            meaningful_chars = len(re.findall(r'[a-zA-Z0-9]', line))
            if meaningful_chars >= 3:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_orphan_page_content(self, doc, page_num: int) -> List[Dict]:
        """Extract content from pages not covered by TOC"""
        page = doc[page_num]
        text = self._extract_best_text(page)
        
        if not text or len(text.strip()) < 50:
            return []
        
        cleaned_text = self._clean_text_advanced(text)
        if not cleaned_text:
            return []
        
        # Try to determine what type of content this is
        title = f"Page {page_num + 1} Content"
        lines = cleaned_text.split('\n')
        if lines:
            # Use first meaningful line as title if it looks like a heading
            first_line = lines[0].strip()
            if len(first_line) < 80 and any(pattern.match(first_line) for pattern in self.heading_patterns):
                title = first_line
        
        return self._create_section_chunks(
            cleaned_text, 
            title, 
            page_num + 1, 
            page_num + 1, 
            0
        )
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Generate embeddings in batches for better performance"""
        print(f"🧠 Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📊 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} items)...")
            
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if batch_num < total_batches:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {e}")
                all_embeddings.extend([None] * len(batch_texts))
        
        successful_embeddings = len([e for e in all_embeddings if e is not None])
        print(f"✅ Generated {successful_embeddings}/{len(all_embeddings)} embeddings successfully")
        return all_embeddings
    
    def process_manual_structured(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Complete structured processing pipeline for racing manual"""
        print(f"\n{'='*70}")
        print(f"🏎️  PROCESSING MANUAL: {manual_id}")
        print(f"{'='*70}")
        
        # Step 1: Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Step 2: Extract table of contents and build section index
        toc_data = self.extract_table_of_contents(pdf_path)
        
        # Step 3: Extract structured content using TOC
        structured_content = self.extract_structured_content(pdf_path, toc_data)
        
        if not structured_content:
            print("❌ No meaningful content extracted!")
            return []
        
        # Step 4: Prepare texts for embedding
        chunk_texts = [chunk["chunk_text"] for chunk in structured_content]
        
        # Step 5: Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(chunk_texts, batch_size=25)
        
        # Step 6: Create final structured chunks with embeddings
        final_chunks = []
        
        for i, (chunk, embedding) in enumerate(zip(structured_content, embeddings)):
            if embedding is None:
                print(f"⚠️  Skipping chunk {i} due to embedding failure")
                continue
            
            final_chunk = {
                "manual_id": manual_id,
                "section_title": chunk["section_title"],
                "chunk_text": chunk["chunk_text"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks_in_section": chunk["total_chunks_in_section"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_level": chunk["section_level"],
                "section_type": chunk["section_type"],
                "word_count": chunk["word_count"],
                "token_count": chunk["token_count"],
                "char_count": chunk["char_count"],
                "structured_data": chunk["structured_data"],
                "embedding_vector": embedding,
                "embedding_dimensions": len(embedding),
                "has_overlap": chunk["has_overlap"],
                "overlap_tokens": chunk["overlap_tokens"]
            }
            
            final_chunks.append(final_chunk)
        
        # Step 7: Print processing summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📄 Total pages processed: {len(set(c['page_start'] for c in final_chunks))}")
        print(f"   📚 TOC sections found: {len(toc_data)}")
        print(f"   🔢 Final chunks created: {len(final_chunks)}")
        
        if final_chunks:
            avg_tokens = sum(c['token_count'] for c in final_chunks) / len(final_chunks)
            checklists = len([c for c in final_chunks if c['section_type'] == 'checklist'])
            procedures = len([c for c in final_chunks if c['section_type'] == 'procedure'])
            
            print(f"   ⚖️  Average tokens per chunk: {avg_tokens:.1f}")
            print(f"   ✅ Checklist sections: {checklists}")
            print(f"   🔧 Procedure sections: {procedures}")
            
            # Show sample structured data
            with_structured = [c for c in final_chunks if c['structured_data']]
            print(f"   📋 Chunks with structured data: {len(with_structured)}")
        
        return final_chunks

def main():
    """Process Radical racing manuals with structured approach"""
    processor = StructuredPDFProcessor(
        max_tokens_per_chunk=400,
        overlap_tokens=50
    )
    
    # Manual configurations
    manuals = [
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
    
    all_structured_chunks = []
    
    for manual_config in manuals:
        chunks = processor.process_manual_structured(
            manual_config["url"],
            manual_config["filename"],
            manual_config["manual_id"]
        )
        
        all_structured_chunks.extend(chunks)
    
    # Save structured chunks
    output_file = "chunks_with_embeddings_structured.json"
    with open(output_file, "w") as f:
        json.dump(all_structured_chunks, f, indent=2)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   📁 Total structured chunks: {len(all_structured_chunks)}")
    print(f"   💾 Saved to: {output_file}")
    
    # Analysis by section type
    section_types = {}
    for chunk in all_structured_chunks:
        section_type = chunk.get('section_type', 'general')
        section_types[section_type] = section_types.get(section_type, 0) + 1
    
    print(f"\n📊 SECTION TYPE BREAKDOWN:")
    for section_type, count in sorted(section_types.items()):
        print(f"   {section_type}: {count} chunks")

if __name__ == "__main__":
    main()