import re
from typing import List, Dict, Any
import fitz  # PyMuPDF

class PDFLoader:
    def __init__(self, doc_id: str, book_type: str, version: str):
        self.doc_id = doc_id
        self.book_type = book_type
        self.version = version

    def load(self, file_path: str, start_page: int = 1, max_pages: int = None) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        pages_data = []
        
        start_index = max(0, start_page - 1)
        
        for page_num in range(start_index, len(doc)):
            pages_read = page_num - start_index
            if max_pages is not None and pages_read >= max_pages:
                break
                
            page = doc[page_num]
            text = page.get_text("text")
            layout_data = page.get_text("blocks")
            pages_data.append({
                "page_num": page_num + 1,
                "text": text,
                "layout": layout_data
            })
        return pages_data
