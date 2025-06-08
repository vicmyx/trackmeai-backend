#!/usr/bin/env python3
"""
Check what content was actually extracted from the PDFs
"""

import fitz  # PyMuPDF
import os

def check_pdf_content():
    """Check what's actually in the PDF files"""
    
    print("🔍 CHECKING PDF CONTENT")
    print("=" * 50)
    
    pdfs = [
        ("pdfs/Radical_SR3_Owners_Manual.pdf", "SR3 Owner's Manual"),
        ("pdfs/Radical_Handling_Guide.pdf", "Handling Guide")
    ]
    
    for pdf_path, name in pdfs:
        if not os.path.exists(pdf_path):
            print(f"❌ {name}: File not found at {pdf_path}")
            continue
        
        print(f"\n📖 {name}")
        print("-" * 30)
        
        try:
            doc = fitz.open(pdf_path)
            print(f"📄 Total pages: {doc.page_count}")
            
            # Check first few pages
            for page_num in range(min(5, doc.page_count)):
                page = doc[page_num]
                text = page.get_text()
                
                print(f"\nPage {page_num + 1}:")
                print(f"  Characters: {len(text)}")
                
                if text.strip():
                    # Show first 200 characters
                    preview = text.strip()[:200].replace('\n', ' ')
                    print(f"  Preview: {preview}...")
                    
                    # Count meaningful words
                    words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
                    print(f"  Meaningful words: {len(words)}")
                else:
                    print("  No text content")
            
            # Check a few middle pages
            middle_start = doc.page_count // 2
            print(f"\nChecking middle pages ({middle_start}-{middle_start+2}):")
            
            for page_num in range(middle_start, min(middle_start + 3, doc.page_count)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
                    preview = text.strip()[:100].replace('\n', ' ')
                    print(f"  Page {page_num + 1}: {len(words)} words - {preview}...")
                else:
                    print(f"  Page {page_num + 1}: No text")
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

if __name__ == "__main__":
    check_pdf_content()