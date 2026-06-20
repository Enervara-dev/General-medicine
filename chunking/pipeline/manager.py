from chunking.loaders.pdf_loader import PDFLoader
from chunking.cleaners.text_cleaner import TextCleaner
from chunking.detectors.structure import StructureDetector
from chunking.extractors.semantic import SemanticExtractor
from chunking.llm.retry_engine import ExtractionWithRetry
from chunking.normalizers.medical import MedicalNormalizer
from chunking.chunkers.sub_chunker import SubChunker
from chunking.storage.versioned_storage import VersionedStorage
from chunking.schemas.models import DocumentMetadata
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.detector = StructureDetector()
        self.semantic_extractor = SemanticExtractor()
        self.extractor = ExtractionWithRetry()
        self.normalizer = MedicalNormalizer()
        self.sub_chunker = SubChunker()
        self.storage = VersionedStorage()

    def process_single_block(self, block, version):
        # Idempotency check: Skip if already processed
        processed_file = Path("logs/processed_blocks") / f"{block.block_id}.done"
        if processed_file.exists():
            return []

        structured_data = self.extractor.run(block.text)
        micro_chunks_generated = []
        
        if structured_data:
            normalized_data = self.normalizer.normalize(structured_data)
            
            for chunk in normalized_data.chunks:
                self.storage.save_chunk(chunk, index=1, version=version)
                micro_chunks_generated.append(chunk)

            # Mark processed to prevent duplicates when retrying
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            processed_file.touch()
        else:
            logger.error(f"Failed to extract structured data for block {block.block_id}")
            
        return micro_chunks_generated

    def process_pdf(self, file_path: str, doc_id: str, book_type: str, version: str, start_page: int = 1, max_pages: int = None):
        logger.info(f"Processing PDF: {file_path}")
        
        loader = PDFLoader(doc_id, book_type, version)
        metadata = DocumentMetadata(
            doc_id=doc_id, book_type=book_type, version=version, source_path=file_path
        )
        raw_pages = loader.load(file_path, start_page=start_page, max_pages=max_pages)
        
        if max_pages and len(raw_pages) >= max_pages:
            logger.info(f"Loaded {len(raw_pages)} pages (limited to {max_pages} pages).")
        else:
            logger.info(f"Loaded {len(raw_pages)} pages starting from page {start_page}.")
        
        for page in raw_pages:
            page['text'] = self.cleaner.normalize(page['text'])
            
        sections = self.detector.segment(raw_pages)
        semantic_blocks = self.semantic_extractor.extract_blocks(sections, metadata)
        
        all_micro_chunks = []
        
        # Parallelize at block level for Throughput Optimization
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_block = {executor.submit(self.process_single_block, block, version): block for block in semantic_blocks}
            for future in as_completed(future_to_block):
                try:
                    chunks = future.result()
                    all_micro_chunks.extend(chunks)
                except Exception as exc:
                    block = future_to_block[future]
                    logger.error(f"Block {block.block_id} generated an exception: {exc}")
                
        self.storage.save_chunks(all_micro_chunks, version=version)
        logger.info(f"Pipeline complete for {doc_id}. Generated {len(all_micro_chunks)} chunks.")
