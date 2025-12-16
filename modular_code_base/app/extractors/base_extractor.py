from abc import ABC, abstractmethod  # Import ABC (Abstract Base Class) and abstractmethod decorator to define interfaces/templates for other classes
from typing import List, Dict  # Type hints: List for arrays, Dict for dictionary/object structures


class BaseExtractor(ABC):  # Define abstract base class (template/blueprint) that all extractor classes must inherit from [web:1]
    """Base class for all extractors"""  # Docstring explaining this is the parent class for all rule extraction implementations
    
    def __init__(self, confidence_threshold: float = 0.7):  # Constructor method called when creating a new extractor instance, with default confidence threshold of 0.7 (70%)
        self.confidence_threshold = confidence_threshold  # Store confidence threshold as instance variable to filter low-confidence rules
    
    @abstractmethod  # Decorator marking this method as required to be implemented by all child classes [web:1]
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:  # Abstract method signature: takes content string and optional metadata dict, returns list of rule dictionaries
        """Extract rules from content"""  # Docstring describing that this method extracts rules/patterns from given content
        pass  # Placeholder implementation (child classes MUST override this with actual extraction logic)
    
    def format_rule(self, text: str, confidence: float, rule_type: str) -> Dict:  # Helper method to standardize how rules are structured across all extractors
        """Standardize rule format"""  # Docstring explaining this creates consistent rule dictionary structure
        return {  # Return a dictionary with standardized keys for storing rule information
            'rule_text': text,  # The actual rule content or description as a string
            'confidence': confidence,  # Numerical confidence score (0.0 to 1.0) indicating how reliable this rule is
            'rule_type': rule_type  # Category/type of the rule (e.g., "lint", "security", "build", "test")
        }
