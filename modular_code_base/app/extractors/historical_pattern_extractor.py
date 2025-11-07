"""Historical pattern extraction from build logs"""
import re
from typing import List, Dict

class HistoricalPatternExtractor:
    """Extracts success/failure patterns from logs"""
    
    def extract_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract success and failure patterns"""
        patterns = {
            'success_patterns': [],
            'failure_patterns': []
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Success patterns
            if any(term in line_lower for term in ['success', 'completed', 'passed', '✅']):
                patterns['success_patterns'].append(f"✅ {line.strip()[:80]}")
            
            # Failure patterns
            if any(term in line_lower for term in ['fail', 'error', 'exception', '❌']):
                patterns['failure_patterns'].append(f"❌ {line.strip()[:80]}")
        
        return {
            'success_patterns': patterns['success_patterns'][:5],
            'failure_patterns': patterns['failure_patterns'][:5]
        }
