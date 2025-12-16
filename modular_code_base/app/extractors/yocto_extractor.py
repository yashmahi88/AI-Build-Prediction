import re  # Import regular expression library for pattern matching in Yocto configuration files and build logs
from typing import List, Dict  # Type hints for function signatures: List for arrays, Dict for dictionaries
from app.extractors.base_extractor import BaseExtractor  # Import parent abstract class that defines the extractor interface


class YoctoExtractor(BaseExtractor):  # Define concrete extractor class that inherits from BaseExtractor to extract Yocto-specific build configurations
    """Extract Yocto-specific configurations"""  # Docstring explaining this extractor finds Yocto/OpenEmbedded build system configurations and recipes
    
    def extract(self, content: str, metadata: Dict = None) -> List[Dict]:  # Implementation of abstract extract method: takes content string, optional metadata, returns list of rule dictionaries
        rules = []  # Initialize empty list to store all extracted Yocto configuration rules
        
        patterns = [  # List of tuples defining regex patterns for Yocto configs, their type labels, and confidence scores
            (r'MACHINE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'MACHINE', 0.95),  # Match MACHINE variable assignment (e.g., MACHINE = "raspberrypi4" or MACHINE ?= "qemux86") with 95% confidence - defines target hardware
            (r'BB_NUMBER_THREADS\s*\??=\s*["\']?(\d+)["\']?', 'THREADS', 0.90),  # Match BB_NUMBER_THREADS variable (e.g., BB_NUMBER_THREADS = "8") with 90% confidence - controls BitBake parallel task execution
            (r'PARALLEL_MAKE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'PARALLEL', 0.90),  # Match PARALLEL_MAKE variable (e.g., PARALLEL_MAKE = "-j 8") with 90% confidence - controls parallel compilation jobs within recipes
            (r'DISTRO\s*\??=\s*["\']?([^\"\'\n]+)["\']?', 'DISTRO', 0.85),  # Match DISTRO variable (e.g., DISTRO = "poky") with 85% confidence - defines the distribution configuration
            (r'bitbake\s+([a-zA-Z0-9_-]+)', 'RECIPE', 0.80),  # Match bitbake command with recipe name (e.g., "bitbake core-image-minimal") with 80% confidence - identifies what's being built
            (r'(local\.conf|bblayers\.conf)', 'CONFIG_FILE', 0.75),  # Match references to key Yocto config files (local.conf or bblayers.conf) with 75% confidence - important configuration files
        ]
        
        for pattern, rule_type, conf in patterns:  # Loop through each pattern tuple to search for Yocto configurations
            matches = re.finditer(pattern, content, re.IGNORECASE)  # Find all non-overlapping matches of this pattern in content (case-insensitive search)
            for match in matches:  # Process each individual match found
                value = match.group(1) if match.groups() else match.group(0)  # Extract captured group (the value) if it exists, otherwise use entire match (for CONFIG_FILE pattern which has no capture group for value)
                rules.append(self.format_rule(f'{rule_type}: {value}', conf, f'YOCTO_{rule_type}'))  # Create standardized rule dict using parent class method with format "MACHINE: raspberrypi4" and rule type like "YOCTO_MACHINE"
        
        return rules  # Return complete list of all extracted Yocto configuration rules
