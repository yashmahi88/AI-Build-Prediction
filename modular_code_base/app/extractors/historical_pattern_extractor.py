"""Historical pattern extraction from build logs"""  # Module docstring describing this file extracts recurring success/failure patterns from CI/CD build logs
import re  # Import regular expression library for pattern matching (currently unused but available for future regex patterns)
from typing import List, Dict  # Type hints for function return types: List for arrays, Dict for dictionary structures


class HistoricalPatternExtractor:  # Define class to analyze historical build logs and identify common success/failure indicators
    """Extracts success/failure patterns from logs"""  # Docstring explaining this class finds patterns that correlate with build outcomes
    
    def extract_patterns(self, content: str) -> Dict[str, List[str]]:  # Method that takes log content as string and returns dictionary containing success and failure pattern lists
        """Extract success and failure patterns"""  # Docstring describing this method identifies and categorizes patterns from logs
        patterns = {  # Initialize dictionary to store two categories of patterns
            'success_patterns': [],  # Empty list that will hold lines indicating successful builds
            'failure_patterns': []  # Empty list that will hold lines indicating failed builds
        }
        
        lines = content.split('\n')  # Split the entire log content into individual lines using newline character as delimiter
        
        for line in lines:  # Loop through each line in the log content
            line_lower = line.lower()  # Convert line to lowercase for case-insensitive keyword matching
            
            # Success patterns
            if any(term in line_lower for term in ['success', 'completed', 'passed', '✅']):  # Check if ANY of these success keywords appear in the lowercase line
                patterns['success_patterns'].append(f"✅ {line.strip()[:80]}")  # Add to success list with checkmark emoji, strip whitespace, and limit to first 80 characters to prevent overly long entries
            
            # Failure patterns
            if any(term in line_lower for term in ['fail', 'error', 'exception', '❌']):  # Check if ANY of these failure keywords appear in the lowercase line
                patterns['failure_patterns'].append(f"❌ {line.strip()[:80]}")  # Add to failure list with X emoji, strip whitespace, and limit to first 80 characters
        
        return {  # Return dictionary with limited pattern lists (prevents overwhelming output)
            'success_patterns': patterns['success_patterns'][:5],  # Take only first 5 success patterns found (slice notation)
            'failure_patterns': patterns['failure_patterns'][:5]  # Take only first 5 failure patterns found (slice notation)
        }
