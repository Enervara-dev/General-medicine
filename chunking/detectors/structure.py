import re
from typing import List, Dict, Any

class StructureDetector:
    def segment(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sections = []
        current_section = "General"
        current_text = []
        
        section_pattern = re.compile(r'^(Symptoms|Diagnosis|Treatment|Introduction|Pathophysiology)\s*$', re.IGNORECASE)

        for page in pages:
            lines = page['text'].split('\n')
            for line in lines:
                match = section_pattern.match(line.strip())
                if match:
                    if current_text:
                        sections.append({
                            "section": current_section,
                            "text": "\n".join(current_text)
                        })
                    current_section = match.group(1).title()
                    current_text = []
                else:
                    current_text.append(line)
        
        if current_text:
            sections.append({
                "section": current_section,
                "text": "\n".join(current_text)
            })
            
        return sections
