import re
import unicodedata

class TextCleaner:
    def normalize(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Remove whitespace noise
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove typical headers/footers and noise
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_s = line.strip()
            
            # Remove isolated tokens (S1, t1, 1, a)
            if re.fullmatch(r'[A-Za-z0-9]{1,2}', line_s):
                continue
            
            # Remove captions / figures / references
            if re.match(r'^(figure|fig\.|table|video|references?)\b', line_s, re.IGNORECASE):
                continue
                
            if re.match(r'^Page \d+$', line_s) or re.match(r'^\d+$', line_s):
                continue
                
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)
