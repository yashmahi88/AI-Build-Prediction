import re
from typing import List, Set, Dict
from app.extractors.base_extractor import BaseExtractor

class TermExtractor(BaseExtractor):
    """Unified technical term extraction - consolidates all term extraction"""
    
    def __init__(self):
        super().__init__()
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
    
    def extract(self, content: str, metadata: Dict = None) -> List[str]:
        """Extract all types of technical terms"""
        terms = set()
        
        # Technical patterns
        terms.update(self._extract_patterns(content))
        
        # Configuration terms
        terms.update(self._extract_config_terms(content))
        
        # Key concepts (3+ chars, no stopwords)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        terms.update([w for w in words if w.lower() not in self.stopwords])
        
        return sorted(list(terms))
    
    def _extract_patterns(self, content: str) -> Set[str]:
        """Extract technical patterns (CONSTANTS, snake_case, camelCase)"""
        patterns = [
            r'\b[A-Z][A-Z_]+\b',
            r'\b\w+_\w+\b',
            r'\b[a-z]+[A-Z]\w+\b'
        ]
        terms = set()
        for pattern in patterns:
            terms.update(re.findall(pattern, content))
        return terms
    
    def _extract_config_terms(self, content: str) -> Set[str]:
        """Extract configuration key names"""
        matches = re.findall(r'([A-Z_]+)\s*\??=', content)
        return set(matches)
