import re  # Import regular expression library for pattern matching in text
from typing import List, Dict  # Type hints for function signatures: List for arrays, Dict for dictionaries
from app.extractors.base_extractor import BaseExtractor  # Import parent abstract class that defines the extractor interface


class ConstraintExtractor(BaseExtractor):  # Define concrete extractor class that inherits from BaseExtractor to extract resource/constraint rules
    """Extract constraints and resource requirements"""  # Docstring explaining this extractor finds resource limits and requirements in text
    
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:  # Implementation of abstract extract method: takes content string, optional metadata, returns list of rule dictionaries
        rules = []  # Initialize empty list to store all extracted constraint rules
        
        patterns = [  # List of tuples defining regex patterns, constraint type labels, and confidence scores
            (r'(\d+)\s*(GB|MB|TB|gb|mb|tb)', 'STORAGE', 0.90),  # Pattern to match storage sizes like "10 GB" or "500 MB" with 90% confidence
            (r'(\d+)\s*(hours?|hrs?|minutes?|mins?)', 'TIME', 0.85),  # Pattern to match time durations like "2 hours" or "30 mins" with 85% confidence
            (r'(\d+)\s*(cores?|threads?|CPUs?)', 'COMPUTE', 0.85),  # Pattern to match CPU/compute resources like "4 cores" or "8 threads" with 85% confidence
            (r'minimum\s+(.+)', 'MINIMUM', 0.80),  # Pattern to match minimum requirements like "minimum 4GB RAM" with 80% confidence
            (r'maximum\s+(.+)', 'MAXIMUM', 0.80),  # Pattern to match maximum limits like "maximum 10 concurrent builds" with 80% confidence
            (r'memory\s+(.+)', 'MEMORY', 0.80),  # Pattern to match memory-related text like "memory 8GB required" with 80% confidence
        ]
        
        for pattern, constraint_type, conf in patterns:  # Loop through each pattern tuple to search for matches
            matches = re.finditer(pattern, content, re.IGNORECASE)  # Find all non-overlapping matches of this pattern in content (case-insensitive search)
            for match in matches:  # Process each individual match found
                context = content[max(0, match.start()-50):min(len(content), match.end()+50)].strip()  # Extract surrounding text: 50 chars before to 50 chars after match (with boundary checking), then remove leading/trailing whitespace
                rules.append(self.format_rule(context, conf, f'CONSTRAINT_{constraint_type}'))  # Create standardized rule dict using parent class method and add to rules list with type like "CONSTRAINT_STORAGE"
        
        return rules  # Return complete list of all extracted constraint rules
