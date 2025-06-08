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

class EnhancedOCRProcessor:
    """Enhanced PDF processor with better OCR and text extraction"""
    
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
        print(f"📥 Downloading {filename}...")
        
        response = requests.get(url)
        response.raise_for_status()
        
        os.makedirs("pdfs", exist_ok=True)
        filepath = f"pdfs/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {filepath}")
        return filepath
    
    def extract_toc_from_handling_guide(self, pdf_path: str) -> Dict[str, Dict]:
        """Extract specific TOC from Radical Handling Guide"""
        doc = fitz.open(pdf_path)
        toc_sections = {}
        
        # We know the TOC structure from our previous analysis
        known_sections = {
            "Introduction": {"page": 4, "type": "general"},
            "Handling & Setup Guide": {"page": 4, "type": "setup"},
            "Some Examples of Driver Feedback": {"page": 5, "type": "general"},
            "Guidelines on the Effects of Various Setup Changes": {"page": 6, "type": "setup"},
            "Stiffer Front Anti-Roll Bar": {"page": 6, "type": "setup"},
            "Stiffer Rear Anti-Roll Bar": {"page": 6, "type": "setup"},
            "Stiffer Front Spring": {"page": 6, "type": "setup"},
            "Stiffer Rear Spring": {"page": 6, "type": "setup"},
            "Increase Camber": {"page": 6, "type": "setup"},
            "Higher Tyre Pressure": {"page": 6, "type": "specification"},
            "Front Toe In/Out": {"page": 7, "type": "setup"},
            "Rear Toe In/Out": {"page": 7, "type": "setup"},
            "Front Ride Height": {"page": 7, "type": "setup"},
            "Rear Ride Height": {"page": 7, "type": "setup"},
            "Wings/Dive Planes": {"page": 7, "type": "setup"},
            "Front Bump Damping": {"page": 7, "type": "setup"},
            "Rear Bump Damping": {"page": 8, "type": "setup"},
            "Front Rebound Damping": {"page": 8, "type": "setup"},
            "Rear Rebound Damping": {"page": 8, "type": "setup"},
            "Increase Front Preload": {"page": 8, "type": "setup"},
            "Handling Issues and Potential Solutions": {"page": 8, "type": "procedure"}
        }
        
        for section_title, info in known_sections.items():
            toc_sections[section_title] = {
                "title": section_title,
                "page": info["page"],
                "type": info["type"],
                "level": 1
            }
        
        doc.close()
        print(f"✅ Loaded {len(toc_sections)} known sections from Handling Guide")
        return toc_sections
    
    def extract_comprehensive_content(self, pdf_path: str) -> List[Dict]:
        """Extract content using multiple methods including OCR-like approaches"""
        doc = fitz.open(pdf_path)
        content_blocks = []
        
        print(f"🔍 Comprehensive extraction from {doc.page_count} pages...")
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Method 1: Standard text extraction
            text1 = page.get_text()
            
            # Method 2: Text blocks extraction
            text2 = self._extract_from_text_blocks(page)
            
            # Method 3: Character-level extraction
            text3 = self._extract_char_level(page)
            
            # Method 4: Dictionary-based extraction with layout preservation
            text4 = self._extract_with_layout(page)
            
            # Choose the best extraction
            texts = [text1, text2, text3, text4]
            best_text = self._choose_best_text(texts)
            
            if best_text and len(best_text.strip()) > 20:
                cleaned_text = self._clean_text_comprehensive(best_text)
                
                if cleaned_text and len(cleaned_text.strip()) > 30:
                    # Create content block for this page
                    content_block = {
                        "page_number": page_num + 1,
                        "raw_text": best_text[:500] + "..." if len(best_text) > 500 else best_text,
                        "cleaned_text": cleaned_text,
                        "extraction_method": self._get_best_method_name(texts, best_text),
                        "content_length": len(cleaned_text),
                        "word_count": len(cleaned_text.split())
                    }
                    content_blocks.append(content_block)
                    print(f"📄 Page {page_num + 1}: {len(cleaned_text)} chars extracted")
        
        doc.close()
        print(f"✅ Extracted content from {len(content_blocks)} pages")
        return content_blocks
    
    def _extract_from_text_blocks(self, page) -> str:
        """Extract using text blocks method"""
        try:
            blocks = page.get_text("blocks")
            text_parts = []
            
            for block in blocks:
                if len(block) > 4 and isinstance(block[4], str):
                    block_text = block[4].strip()
                    if block_text:
                        text_parts.append(block_text)
            
            return "\n".join(text_parts)
        except:
            return ""
    
    def _extract_char_level(self, page) -> str:
        """Extract using character-level analysis"""
        try:
            text_dict = page.get_text("dict")
            chars = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            line_chars = []
                            for span in line["spans"]:
                                text = span.get("text", "")
                                if text.strip():
                                    line_chars.append(text)
                            if line_chars:
                                chars.append("".join(line_chars))
            
            return "\n".join(chars)
        except:
            return ""
    
    def _extract_with_layout(self, page) -> str:
        """Extract preserving layout structure"""
        try:
            text_dict = page.get_text("dict")
            lines_with_position = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            line_text = ""
                            line_bbox = line.get("bbox", [0, 0, 0, 0])
                            
                            for span in line["spans"]:
                                text = span.get("text", "")
                                if text.strip():
                                    line_text += text
                            
                            if line_text.strip():
                                lines_with_position.append({
                                    "text": line_text.strip(),
                                    "y": line_bbox[1],  # Y position for sorting
                                    "x": line_bbox[0]   # X position
                                })
            
            # Sort by Y position (top to bottom)
            lines_with_position.sort(key=lambda x: x["y"])
            
            # Extract text maintaining order
            ordered_text = [line["text"] for line in lines_with_position]
            return "\n".join(ordered_text)
            
        except:
            return ""
    
    def _choose_best_text(self, texts: List[str]) -> str:
        """Choose the best extraction from multiple methods"""
        if not texts:
            return ""
        
        # Score each text based on content quality
        scored_texts = []
        
        for text in texts:
            if not text:
                continue
            
            score = 0
            # More meaningful words = higher score
            words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
            score += len(words) * 2
            
            # Longer text usually better (up to a point)
            score += min(len(text) / 10, 100)
            
            # Penalize texts with too many repeated characters
            if len(set(text)) < len(text) * 0.1:
                score -= 50
            
            # Bonus for setup-related keywords
            setup_keywords = ['pressure', 'camber', 'toe', 'spring', 'damping', 'height', 'roll', 'bar']
            for keyword in setup_keywords:
                if keyword.lower() in text.lower():
                    score += 10
            
            scored_texts.append((score, text))
        
        if not scored_texts:
            return ""
        
        # Return highest scoring text
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        return scored_texts[0][1]
    
    def _get_best_method_name(self, texts: List[str], best_text: str) -> str:
        """Identify which extraction method produced the best text"""
        methods = ["standard", "blocks", "char_level", "layout"]
        
        for i, text in enumerate(texts):
            if text == best_text:
                return methods[i] if i < len(methods) else "unknown"
        
        return "unknown"
    
    def _clean_text_comprehensive(self, text: str) -> str:
        """Comprehensive text cleaning"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip pure page numbers
            if re.match(r'^\d+$', line) and len(line) < 3:
                continue
            
            # Skip lines that are mostly dots or spaces
            if re.match(r'^[.\s]+$', line):
                continue
            
            # Skip very short lines that don't look like content
            if len(line) < 2:
                continue
            
            # Keep lines with meaningful content
            # Must have at least some alphanumeric characters
            if re.search(r'[a-zA-Z0-9]', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_targeted_chunks(self, content_blocks: List[Dict], toc_sections: Dict = None) -> List[Dict]:
        """Create chunks targeting specific manual sections"""
        chunks = []
        
        # If we have TOC data, try to match content to sections
        if toc_sections:
            page_to_sections = {}
            for section_title, section_info in toc_sections.items():
                page_num = section_info["page"]
                if page_num not in page_to_sections:
                    page_to_sections[page_num] = []
                page_to_sections[page_num].append((section_title, section_info))
        
        for content_block in content_blocks:
            page_num = content_block["page_number"]
            text = content_block["cleaned_text"]
            
            if not text or len(text.strip()) < 20:
                continue
            
            # Determine section information
            section_title = f"Page {page_num}"
            section_type = "general"
            
            if toc_sections and page_num in page_to_sections:
                # Try to match content to specific sections on this page
                page_sections = page_to_sections[page_num]
                
                if len(page_sections) == 1:
                    # Only one section on this page
                    section_title = page_sections[0][0]
                    section_type = page_sections[0][1]["type"]
                else:
                    # Multiple sections - try to split content
                    section_chunks = self._split_content_by_sections(text, page_sections)
                    for section_chunk in section_chunks:
                        chunk = self._create_chunk(
                            section_chunk["text"],
                            section_chunk["title"],
                            page_num,
                            section_chunk["type"],
                            content_block
                        )
                        chunks.append(chunk)
                    continue
            
            # Create single chunk for this page/section
            chunk = self._create_chunk(text, section_title, page_num, section_type, content_block)
            chunks.append(chunk)
        
        return chunks
    
    def _split_content_by_sections(self, text: str, page_sections: List[Tuple]) -> List[Dict]:
        """Split page content into sections based on headings"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line matches any section title
            matched_section = None
            for section_title, section_info in page_sections:
                # Try exact match or partial match
                if (line.lower() == section_title.lower() or 
                    section_title.lower() in line.lower() or
                    line.lower() in section_title.lower()):
                    matched_section = (section_title, section_info)
                    break
            
            if matched_section:
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        "title": current_section[0],
                        "type": current_section[1]["type"],
                        "text": '\n'.join(current_content)
                    })
                
                # Start new section
                current_section = matched_section
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections.append({
                "title": current_section[0],
                "type": current_section[1]["type"],
                "text": '\n'.join(current_content)
            })
        
        # If no sections were matched, return original as one section
        if not sections and page_sections:
            sections.append({
                "title": page_sections[0][0],
                "type": page_sections[0][1]["type"],
                "text": text
            })
        
        return sections
    
    def _create_chunk(self, text: str, section_title: str, page_num: int, section_type: str, content_block: Dict) -> Dict:
        """Create a chunk with all metadata"""
        if not text.strip():
            return None
        
        token_count = self.count_tokens(text)
        word_count = len(text.split())
        
        # Extract structured data based on section type
        structured_data = {}
        if section_type == "setup":
            structured_data = self._extract_setup_parameters(text)
        elif section_type == "specification":
            structured_data = self._extract_specifications(text)
        elif section_type == "procedure":
            structured_data = self._extract_procedure_steps(text)
        
        return {
            "section_title": section_title,
            "chunk_text": text,
            "page_start": page_num,
            "page_end": page_num,
            "section_type": section_type,
            "word_count": word_count,
            "token_count": token_count,
            "char_count": len(text),
            "extraction_method": content_block["extraction_method"],
            "structured_data": structured_data
        }
    
    def _extract_setup_parameters(self, text: str) -> Dict:
        """Extract setup parameters and their effects"""
        parameters = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for parameter descriptions
            # Example: "Higher Tyre Pressure: Increases understeer"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    param = parts[0].strip()
                    effect = parts[1].strip()
                    parameters[param] = effect
            
            # Look for pressure values
            pressure_match = re.search(r'(\d+)\s*psi', line.lower())
            if pressure_match:
                parameters["tire_pressure_psi"] = int(pressure_match.group(1))
        
        return {"setup_parameters": parameters} if parameters else {}
    
    def _extract_specifications(self, text: str) -> Dict:
        """Extract technical specifications"""
        specs = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for pressure specifications
            if 'pressure' in line.lower():
                # Extract tire pressures
                hankook_match = re.search(r'hankook.*?(\d+)\s*psi', line.lower())
                dunlop_match = re.search(r'dunlop.*?(\d+)\s*psi', line.lower())
                
                if hankook_match:
                    specs["hankook_pressure_psi"] = int(hankook_match.group(1))
                if dunlop_match:
                    specs["dunlop_pressure_psi"] = int(dunlop_match.group(1))
            
            # Look for other numeric specifications
            numeric_match = re.search(r'(\w+):\s*(\d+(?:\.\d+)?)\s*(\w+)', line)
            if numeric_match:
                param = numeric_match.group(1).lower()
                value = float(numeric_match.group(2))
                unit = numeric_match.group(3)
                specs[f"{param}_{unit}"] = value
        
        return {"specifications": specs} if specs else {}
    
    def _extract_procedure_steps(self, text: str) -> Dict:
        """Extract procedure steps"""
        steps = []
        lines = text.split('\n')
        
        step_number = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered steps or bullet points
            if re.match(r'^\d+\.', line) or re.match(r'^[-*•]', line):
                steps.append({
                    "step_number": step_number,
                    "instruction": line,
                    "type": "step"
                })
                step_number += 1
            elif len(steps) > 0:
                # Add as detail to last step
                if "details" not in steps[-1]:
                    steps[-1]["details"] = []
                steps[-1]["details"].append(line)
        
        return {"procedure_steps": steps} if steps else {}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """Generate embeddings in batches"""
        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        
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
        
        successful = len([e for e in all_embeddings if e is not None])
        print(f"✅ Generated {successful}/{len(all_embeddings)} embeddings successfully")
        return all_embeddings
    
    def process_manual_enhanced(self, url: str, filename: str, manual_id: str) -> List[Dict]:
        """Enhanced processing pipeline"""
        print(f"\n{'='*70}")
        print(f"🏎️  ENHANCED PROCESSING: {manual_id}")
        print(f"{'='*70}")
        
        # Download PDF
        pdf_path = self.download_pdf(url, filename)
        
        # Extract TOC if it's the Handling Guide
        toc_sections = {}
        if "handling" in manual_id.lower():
            toc_sections = self.extract_toc_from_handling_guide(pdf_path)
        
        # Extract comprehensive content
        content_blocks = self.extract_comprehensive_content(pdf_path)
        
        if not content_blocks:
            print("❌ No content extracted!")
            return []
        
        # Create targeted chunks
        chunks = self.create_targeted_chunks(content_blocks, toc_sections)
        
        if not chunks:
            print("❌ No chunks created!")
            return []
        
        # Generate embeddings
        chunk_texts = [chunk["chunk_text"] for chunk in chunks if chunk]
        embeddings = self.generate_embeddings_batch(chunk_texts)
        
        # Create final chunks with embeddings
        final_chunks = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not chunk or embedding is None:
                continue
            
            final_chunk = {
                "manual_id": manual_id,
                "section_title": chunk["section_title"],
                "chunk_text": chunk["chunk_text"],
                "chunk_index": i,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_type": chunk["section_type"],
                "word_count": chunk["word_count"],
                "token_count": chunk["token_count"],
                "char_count": chunk["char_count"],
                "extraction_method": chunk["extraction_method"],
                "structured_data": chunk["structured_data"],
                "embedding_vector": embedding,
                "embedding_dimensions": len(embedding)
            }
            
            final_chunks.append(final_chunk)
        
        # Print summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📄 Content blocks extracted: {len(content_blocks)}")
        print(f"   📚 TOC sections: {len(toc_sections)}")
        print(f"   🔢 Final chunks: {len(final_chunks)}")
        
        if final_chunks:
            avg_tokens = sum(c['token_count'] for c in final_chunks) / len(final_chunks)
            setup_chunks = len([c for c in final_chunks if c['section_type'] == 'setup'])
            spec_chunks = len([c for c in final_chunks if c['section_type'] == 'specification'])
            
            print(f"   ⚖️  Average tokens per chunk: {avg_tokens:.1f}")
            print(f"   🔧 Setup sections: {setup_chunks}")
            print(f"   📋 Specification sections: {spec_chunks}")
            
            # Show extraction methods used
            methods = {}
            for chunk in final_chunks:
                method = chunk.get('extraction_method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
            
            print(f"   🔍 Extraction methods used:")
            for method, count in methods.items():
                print(f"      {method}: {count} chunks")
        
        return final_chunks

def main():
    """Process manuals with enhanced OCR extraction"""
    processor = EnhancedOCRProcessor(
        max_tokens_per_chunk=400,
        overlap_tokens=50
    )
    
    # Process manuals
    manuals = [
        {
            "url": "https://qyihtrkdkcsxzprvbtyy.supabase.co/storage/v1/object/public/files//Radical_Handling_Guide_Master_v1.1.pdf",
            "filename": "Radical_Handling_Guide.pdf",
            "manual_id": "Radical_Handling_Guide"
        },
        {
            "url": "https://qyihtrkdkcsxzprvbtyy.supabase.co/storage/v1/object/public/files//QD17010_Radical_SR3_XXR_Owners_Manual.pdf",
            "filename": "Radical_SR3_Owners_Manual.pdf",
            "manual_id": "Radical_SR3_Owners_Manual"
        }
    ]
    
    all_enhanced_chunks = []
    
    for manual_config in manuals:
        chunks = processor.process_manual_enhanced(
            manual_config["url"],
            manual_config["filename"],
            manual_config["manual_id"]
        )
        
        all_enhanced_chunks.extend(chunks)
    
    # Save enhanced chunks
    output_file = "chunks_with_embeddings_enhanced_ocr.json"
    with open(output_file, "w") as f:
        json.dump(all_enhanced_chunks, f, indent=2)
    
    print(f"\n🎯 FINAL ENHANCED RESULTS:")
    print(f"   📁 Total enhanced chunks: {len(all_enhanced_chunks)}")
    print(f"   💾 Saved to: {output_file}")
    
    # Analysis
    section_types = {}
    extraction_methods = {}
    
    for chunk in all_enhanced_chunks:
        section_type = chunk.get('section_type', 'general')
        section_types[section_type] = section_types.get(section_type, 0) + 1
        
        method = chunk.get('extraction_method', 'unknown')
        extraction_methods[method] = extraction_methods.get(method, 0) + 1
    
    print(f"\n📊 SECTION TYPE BREAKDOWN:")
    for section_type, count in sorted(section_types.items()):
        print(f"   {section_type}: {count} chunks")
    
    print(f"\n🔍 EXTRACTION METHOD BREAKDOWN:")
    for method, count in sorted(extraction_methods.items()):
        print(f"   {method}: {count} chunks")

if __name__ == "__main__":
    main()