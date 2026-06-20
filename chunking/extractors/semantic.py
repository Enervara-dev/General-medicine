from typing import List
from chunking.schemas.models import SemanticBlock, DocumentMetadata

class SemanticExtractor:
    def extract_blocks(self, sections: List[dict], metadata: DocumentMetadata) -> List[SemanticBlock]:
        blocks = []
        block_counter = 1
        
        for p_sec in sections:
            section_name = p_sec['section']
            text = p_sec['text']
            
            # Split into clinically coherent units based on semantic boundaries
            paragraphs = text.split('\n\n')
            current_merged_text = []
            
            import re
            concept_headers = re.compile(
                r'^(treatment|diagnosis|pathophysiology|etiology|clinical features|management|epidemiology|prognosis|pathogenesis|history|physical examination|complications|prevention|indications|contraindications)\b', 
                re.IGNORECASE
            )
            
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 20: 
                    continue
                
                # Split when: new concept starts, drug introduced, disease mentioned
                is_boundary = False
                if concept_headers.match(para):
                    is_boundary = True
                # Short phrase ending in colon/period often introduces a new disease/drug
                elif re.match(r'^([A-Z][A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+){0,3})\s*[:\.]', para): 
                    is_boundary = True
                # Lists introducing independent drugs or disease types
                elif para.startswith('\u2022') or para.startswith('- '):
                    is_boundary = True
                    
                # Explicit constraints: if tokens > 350: split()
                approx_para_tokens = len(para.split()) * 1.3
                approx_current_tokens = len("\n\n".join(current_merged_text).split()) * 1.3
                if approx_current_tokens + approx_para_tokens > 350:
                    is_boundary = True
                
                if is_boundary and current_merged_text:
                    blocks.append(SemanticBlock(
                        block_id=f"{metadata.doc_id}-blk-{block_counter}",
                        text="\n\n".join(current_merged_text),
                        section=section_name,
                        metadata=metadata
                    ))
                    block_counter += 1
                    current_merged_text = []
                    
                current_merged_text.append(para)
                
            if current_merged_text:
                blocks.append(SemanticBlock(
                    block_id=f"{metadata.doc_id}-blk-{block_counter}",
                    text="\n\n".join(current_merged_text),
                    section=section_name,
                    metadata=metadata
                ))
                block_counter += 1
                
        return blocks
