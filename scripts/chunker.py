import logging
import os
from chunking.pipeline.manager import DocumentProcessingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_chunking():
    pipeline = DocumentProcessingPipeline()
    docs_dir = "documents"
    
    if not os.path.exists(docs_dir):
        logging.warning(f"No documents folder found at {docs_dir}.")
        return

    # Iterate categories in documents
    categories = [d for d in os.listdir(docs_dir) if os.path.isdir(os.path.join(docs_dir, d))]
    
    if not categories:
        logging.warning(f"No category folders found in {docs_dir}.")
        return

    for category in categories:
        category_dir = os.path.join(docs_dir, category)
        books = [f for f in os.listdir(category_dir) if f.lower().endswith('.pdf')]
        
        for book in books:
            book_path = os.path.join(category_dir, book)
            # Create a doc_id from book name
            doc_id = book.replace(" ", "_").replace(".pdf", "").lower()
            
            logging.info(f"Starting chunking pipeline for Category: {category} -> Book: {book}")
            
            start_page = 44 if "harrison" in book.lower() else 1
            max_pages = None if "harrison" in book.lower() else 5

            pipeline.process_pdf(
                file_path=book_path,
                doc_id=doc_id,
                book_type=category.replace("_", " ").title(),
                version="v1",
                start_page=start_page,
                max_pages=max_pages
            )

    logging.info("Completed chunking process across categories.")

if __name__ == "__main__":
    run_chunking()
