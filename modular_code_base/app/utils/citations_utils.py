import re
from typing import Dict, Optional

def build_citation(metadata: Dict) -> Optional[str]:
    """Build citation from document metadata - Confluence or Jenkins sources"""
    if not metadata:
        return None
    
    confluence_url = metadata.get('confluence_url', '')
    if confluence_url:
        source = metadata.get('source', '')
        page_name = source.split('/')[-1].replace('.md', '').replace('_', ' ') if '/' in source else 'Confluence'
        return f"{page_name} - {confluence_url}"
    
    source = metadata.get('source', '')
    if 'jenkins' in source.lower():
        match = re.search(r'([a-zA-Z0-9_-]+)-(\d+)', source)
        if match:
            return f"Jenkins: {match.group(1)} #{match.group(2)} - http://localhost:8080/job/{match.group(1)}/{match.group(2)}"
    
    return source if source else None
