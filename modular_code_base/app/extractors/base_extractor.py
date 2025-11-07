from abc import ABC, abstractmethod
from typing import List, Dict

class BaseExtractor(ABC):
    """Base class for all extractors"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:
        """Extract rules from content"""
        pass
    
    def format_rule(self, text: str, confidence: float, rule_type: str) -> Dict:
        """Standardize rule format"""
        return {
            'rule_text': text,
            'confidence': confidence,
            'rule_type': rule_type
        }
