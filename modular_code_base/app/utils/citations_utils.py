import re  # Regular expression library for pattern matching to extract job names and build numbers from Jenkins sources
from typing import Dict, Optional  # Type hints for function signatures: Dict for dictionaries, Optional for nullable return types


def build_citation(metadata: Dict) -> Optional[str]:  # Function to construct human-readable citation string from document metadata (returns None if no citation can be built)
    """Build citation from document metadata - Confluence or Jenkins sources"""  # Docstring explaining this creates formatted citations for Confluence pages or Jenkins builds
    if not metadata:  # Check if metadata dictionary is None or empty
        return None  # Return None if no metadata is available (can't build citation without metadata)
    
    confluence_url = metadata.get('confluence_url', '')  # Extract confluence_url field from metadata (empty string if missing)
    if confluence_url:  # Check if document came from Confluence (has confluence_url)
        source = metadata.get('source', '')  # Extract source field from metadata (contains file path or page identifier)
        page_name = source.split('/')[-1].replace('.md', '').replace('_', ' ') if '/' in source else 'Confluence'  # Extract page name from source path: split by slash, take last element, remove .md extension, replace underscores with spaces (if no slash, default to 'Confluence')
        return f"{page_name} - {confluence_url}"  # Return formatted citation: "Page Name - https://confluence.example.com/page/123"
    
    source = metadata.get('source', '')  # Extract source field again (for non-Confluence sources)
    if 'jenkins' in source.lower():  # Check if source contains 'jenkins' (case-insensitive check to identify Jenkins build logs)
        match = re.search(r'([a-zA-Z0-9_-]+)-(\d+)', source)  # Regex to extract job name and build number from source string (pattern matches "job-name-123" format)
        if match:  # Check if regex found a match (successfully extracted job and build number)
            return f"Jenkins: {match.group(1)} #{match.group(2)} - http://localhost:8080/job/{match.group(1)}/{match.group(2)}"  # Return formatted Jenkins citation with job name, build number, and clickable URL (group(1) is job name, group(2) is build number)
    
    return source if source else None  # Return raw source field as fallback citation if it exists, otherwise None (handles generic sources that aren't Confluence or Jenkins)
