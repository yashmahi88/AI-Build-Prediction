import re
from typing import List, Dict
from app.extractors.base_extractor import BaseExtractor

class YoctoExtractor(BaseExtractor):
    """Extract Yocto-specific configurations"""
    
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:
        rules = []
        
        patterns = [
            (r'MACHINE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'MACHINE', 0.95),
            (r'BB_NUMBER_THREADS\s*\??=\s*["\']?(\d+)["\']?', 'THREADS', 0.90),
            (r'PARALLEL_MAKE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'PARALLEL', 0.90),
            (r'DISTRO\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'DISTRO', 0.85),
            (r'bitbake\s+([a-zA-Z0-9_-]+)', 'RECIPE', 0.80),
            (r'(local\.conf|bblayers\.conf)', 'CONFIG_FILE', 0.75),
        ]
        
        for pattern, rule_type, conf in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                rules.append(self.format_rule(f'{rule_type}: {value}', conf, f'YOCTO_{rule_type}'))
        
        return rules
