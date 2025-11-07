import re
from typing import List, Dict
from app.extractors.base_extractor import BaseExtractor

class ConstraintExtractor(BaseExtractor):
    """Extract constraints and resource requirements"""
    
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:
        rules = []
        
        patterns = [
            (r'(\d+)\s*(GB|MB|TB|gb|mb|tb)', 'STORAGE', 0.90),
            (r'(\d+)\s*(hours?|hrs?|minutes?|mins?)', 'TIME', 0.85),
            (r'(\d+)\s*(cores?|threads?|CPUs?)', 'COMPUTE', 0.85),
            (r'minimum\s+(.+)', 'MINIMUM', 0.80),
            (r'maximum\s+(.+)', 'MAXIMUM', 0.80),
            (r'memory\s+(.+)', 'MEMORY', 0.80),
        ]
        
        for pattern, constraint_type, conf in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                context = content[max(0, match.start()-50):min(len(content), match.end()+50)].strip()
                rules.append(self.format_rule(context, conf, f'CONSTRAINT_{constraint_type}'))
        
        return rules
