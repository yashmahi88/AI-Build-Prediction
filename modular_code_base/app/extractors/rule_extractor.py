"""Rule extraction for Yocto builds"""  # Module docstring describing this file extracts various types of rules from documentation/logs for Yocto build systems
import logging  # Standard Python logging library to track execution and debug information
import re  # Regular expression library for pattern matching in text
from typing import Dict, List  # Type hints for function signatures: Dict for dictionaries, List for arrays


logger = logging.getLogger(__name__)  # Create logger instance for this module to output debug/info/error messages



class RuleExtractor:  # Main class that extracts different types of rules from content (documentation, build logs, etc.)
    """Extract rules from content"""  # Docstring explaining this class identifies and categorizes rules from text
    
    def __init__(self):  # Constructor method called when creating a new RuleExtractor instance (no parameters needed)
        logger.debug("[INIT] RuleExtractor initialized")  # Log debug message indicating the extractor was successfully created
    
    def extract_all_rules(self, content: str) -> List[Dict]:  # Main method that orchestrates extraction of all rule types from given content string
        """Extract all types of rules from content"""  # Docstring explaining this method is the entry point for complete rule extraction
        if not content:  # Check if content is empty, None, or falsy
            return []  # Return empty list immediately if there's nothing to process
        
        all_rules = []  # Initialize empty list to collect rules from all extraction methods
        
        # Extract all rule types
        all_rules.extend(self.extract_linguistic_rules(content))  # Add linguistic pattern rules (must/should/may statements) to the list
        all_rules.extend(self.extract_structural_rules(content))  # Add rules found in document structure (headers, sections) to the list
        all_rules.extend(self.extract_procedural_rules(content))  # Add step-by-step procedural rules (numbered lists, bullets) to the list
        all_rules.extend(self.extract_constraint_rules(content))  # Add constraint/limit rules (storage, time, compute) to the list
        all_rules.extend(self.extract_dependency_rules(content))  # Add dependency/prerequisite rules (requires, depends on) to the list
        
        # Deduplicate
        unique_rules = self._deduplicate(all_rules)  # Remove duplicate rules by comparing rule text to avoid redundancy
        
        return unique_rules  # Return final deduplicated list of all extracted rules
    
    def extract_linguistic_rules(self, content: str) -> List[Dict]:  # Extract rules based on language patterns like "must", "should", "ensure", etc.
        """Extract rules based on linguistic patterns"""  # Docstring describing this method finds rules using keywords and phrases
        rules = []  # Initialize empty list to store extracted linguistic rules
        lines = content.split('\n')  # Split content into individual lines using newline as delimiter
        
        requirement_patterns = [  # List of tuples defining regex patterns for requirement levels and their classification
            (r'\bmust\b', 'MANDATORY'),  # Match word "must" (with word boundaries \b) and classify as mandatory requirement
            (r'\bshall\b', 'MANDATORY'),  # Match "shall" and classify as mandatory (formal/legal language)
            (r'\brequired?\b', 'MANDATORY'),  # Match "require" or "required" and classify as mandatory
            (r'\bmandatory\b', 'MANDATORY'),  # Match "mandatory" explicitly
            (r'\bshould\b', 'RECOMMENDED'),  # Match "should" and classify as recommended (not strictly required)
            (r'\brecommended?\b', 'RECOMMENDED'),  # Match "recommend" or "recommended"
            (r'\bpreferred?\b', 'RECOMMENDED'),  # Match "prefer" or "preferred" as recommended
            (r'\bmay\b', 'OPTIONAL'),  # Match "may" indicating optional behavior
            (r'\boptional\b', 'OPTIONAL'),  # Match "optional" explicitly
            (r'\bcould\b', 'OPTIONAL')  # Match "could" as optional suggestion
        ]
        
        action_patterns = [  # List of tuples defining regex patterns for action verbs and their categories
            (r'\bensure\b', 'VERIFICATION'),  # Match "ensure" as verification action
            (r'\bverify\b', 'VERIFICATION'),  # Match "verify" as verification action
            (r'\bvalidate\b', 'VALIDATION'),  # Match "validate" as validation action (checking correctness)
            (r'\bconfigure\b', 'CONFIGURATION'),  # Match "configure" as configuration action (setting up parameters)
            (r'\bsetup\b', 'SETUP'),  # Match "setup" as initial setup action
            (r'\binstall\b', 'INSTALLATION'),  # Match "install" as installation action
            (r'\bdeploy\b', 'DEPLOYMENT'),  # Match "deploy" as deployment action
            (r'\bbuild\b', 'BUILD'),  # Match "build" as build/compilation action
            (r'\btest\b', 'TESTING'),  # Match "test" as testing action
            (r'\bmonitor\b', 'MONITORING')  # Match "monitor" as monitoring/observability action
        ]
        
        for line in lines:  # Loop through each line of content
            line_stripped = line.strip()  # Remove leading and trailing whitespace from line
            if len(line_stripped) < 10 or len(line_stripped) > 200:  # Skip lines that are too short (noise) or too long (likely not a single rule)
                continue  # Move to next line
                
            line_lower = line_stripped.lower()  # Convert line to lowercase for case-insensitive pattern matching
            
            requirement_level = 'UNKNOWN'  # Default requirement level if no pattern matches
            for pattern, level in requirement_patterns:  # Loop through requirement pattern tuples
                if re.search(pattern, line_lower):  # Search for pattern in lowercase line (case-insensitive)
                    requirement_level = level  # Set the requirement level based on matched pattern
                    break  # Stop checking other patterns once we find a match
            
            action_type = 'GENERAL'  # Default action type if no pattern matches
            for pattern, action in action_patterns:  # Loop through action pattern tuples
                if re.search(pattern, line_lower):  # Search for action pattern in lowercase line
                    action_type = action  # Set the action type based on matched pattern
                    break  # Stop checking other patterns after first match
            
            if requirement_level != 'UNKNOWN' or action_type != 'GENERAL':  # Only add rule if we found at least one meaningful pattern
                rules.append({  # Add rule dictionary to list
                    'rule_text': line_stripped,  # Store the original line text (not lowercased)
                    'requirement_level': requirement_level,  # Store whether it's MANDATORY, RECOMMENDED, OPTIONAL, or UNKNOWN
                    'action_type': action_type,  # Store what action category it belongs to (VERIFICATION, BUILD, etc.)
                    'rule_type': 'LINGUISTIC',  # Mark this as a linguistically-extracted rule
                    'confidence': 0.8  # Assign 80% confidence score to linguistic pattern matches
                })
        
        return rules  # Return list of all linguistic rules found
    
    def extract_structural_rules(self, content: str) -> List[Dict]:  # Extract rules based on document structure like headers and sections
        """Extract rules from document structure"""  # Docstring explaining this method uses document organization to find rules
        rules = []  # Initialize empty list for structural rules
        lines = content.split('\n')  # Split content into individual lines
        
        current_section = ""  # Track the current section header we're under (empty initially)
        section_patterns = [  # List of regex patterns to identify section headers in documentation
            r'^#+\s+(.+)$',    # Markdown headers like "# Section Name" or "## Subsection"
            r'^(.+):$',    # Lines ending with colon like "Requirements:" indicating a section start
            r'^\d+\.\s+(.+)$',  # Numbered headings like "1. Introduction" or "2. Setup"
            r'^[A-Z\s]+$'   # ALL CAPS lines often used as section titles in plain text docs
        ]
        
        requirement_keywords = [  # List of keywords that indicate a section contains rules/requirements
            'requirement', 'prerequisite', 'configuration', 'setup', 'installation',  # Common requirement-related terms
            'environment', 'dependency', 'constraint', 'rule', 'standard', 'guideline'  # More requirement-related terms
        ]
        
        for line in lines:  # Loop through each line in content
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue  # Move to next line
                
            for pattern in section_patterns:  # Check if this line is a section header
                match = re.match(pattern, line)  # Try to match section header pattern from start of line
                if match and len(match.group(1)) < 100:  # If matched and captured group is reasonable length (not overly long)
                    current_section = match.group(1).strip()  # Update current section to the header text
                    break  # Stop checking other patterns once we find a header
            
            if current_section and any(keyword in current_section.lower() for keyword in requirement_keywords):  # If we're in a requirements-related section
                if line != current_section and len(line) > 20:  # If line is not the header itself and is long enough to be meaningful
                    rules.append({  # Add this line as a structural rule
                        'rule_text': line,  # Store the rule text
                        'section': current_section,  # Store which section it came from for context
                        'rule_type': 'STRUCTURAL',  # Mark as structurally-extracted rule
                        'confidence': 0.75  # Assign 75% confidence to structural extraction
                    })
        
        return rules  # Return list of all structural rules found
    
    def extract_procedural_rules(self, content: str) -> List[Dict]:  # Extract step-by-step procedural rules from content
        """Extract procedural rules"""  # Docstring explaining this method finds sequential steps and procedures
        rules = []  # Initialize empty list for procedural rules
        
        step_patterns = [  # List of regex patterns that identify step-by-step instructions
            r'^\d+\.\s+(.+)$',  # Numbered steps like "1. First step" or "2. Second step"
            r'^step\s+\d+[:\-]\s*(.+)$',  # Explicit step format like "Step 1: Do this" or "Step 2- Do that"
            r'^\w+\)\s+(.+)$',  # Lettered/numbered steps like "a) First step" or "1) Another step"
            r'^•\s+(.+)$',  # Bullet points using bullet character
            r'^-\s+(.+)$',  # Bullet points using dash/hyphen
            r'^\*\s+(.+)$'  # Bullet points using asterisk
        ]
        
        lines = content.split('\n')  # Split content into lines
        for line in lines:  # Loop through each line
            line_stripped = line.strip()  # Remove whitespace
            if len(line_stripped) < 10:  # Skip very short lines that are unlikely to be meaningful steps
                continue  # Move to next line
                
            for pattern in step_patterns:  # Check if this line matches any step pattern
                match = re.match(pattern, line_stripped)  # Try to match pattern from start of line
                if match:  # If a step pattern matched
                    step_text = match.group(1).strip()  # Extract the actual step text (without the number/bullet)
                    if len(step_text) > 15:  # Only add if step text is substantial (>15 chars)
                        rules.append({  # Add procedural rule
                            'rule_text': step_text,  # Store the step text
                            'rule_type': 'PROCEDURAL',  # Mark as procedural rule
                            'confidence': 0.65  # Assign 65% confidence (lower than linguistic/structural since format is less specific)
                        })
                    break  # Stop checking other patterns once we find a match
        
        return rules  # Return list of all procedural rules
    
    def extract_constraint_rules(self, content: str) -> List[Dict]:  # Extract resource constraints and limits from content
        """Extract constraints and limits"""  # Docstring explaining this method finds numerical constraints like storage/memory/time limits
        rules = []  # Initialize empty list for constraint rules
        
        constraint_patterns = [  # List of tuples with regex patterns and their constraint type classifications
            (r'(\d+)\s*(GB|MB|TB|gb|mb|tb)', 'STORAGE_CONSTRAINT'),  # Match storage sizes like "500 GB" or "10 TB"
            (r'(\d+)\s*(hours?|minutes?|seconds?|hrs?|mins?|secs?)', 'TIME_CONSTRAINT'),  # Match time durations like "2 hours" or "30 mins"
            (r'(\d+)\s*(cores?|threads?|CPUs?|processors?)', 'COMPUTE_CONSTRAINT'),  # Match CPU resources like "4 cores" or "8 threads"
            (r'(\d+)\s*(MB|GB|TB)\s*(RAM|memory)', 'MEMORY_CONSTRAINT'),  # Match memory like "8 GB RAM" or "16 GB memory"
            (r'minimum\s+(\d+)', 'MINIMUM_REQUIREMENT'),  # Match minimum requirements like "minimum 4"
            (r'maximum\s+(\d+)', 'MAXIMUM_LIMIT'),  # Match maximum limits like "maximum 10"
            (r'at\s+least\s+(\d+)', 'MINIMUM_REQUIREMENT'),  # Match "at least 5" as minimum requirement
            (r'no\s+more\s+than\s+(\d+)', 'MAXIMUM_LIMIT')  # Match "no more than 8" as maximum limit
        ]
        
        for pattern, constraint_type in constraint_patterns:  # Loop through constraint pattern tuples
            matches = re.finditer(pattern, content, re.IGNORECASE)  # Find all occurrences of pattern in content (case-insensitive)
            for match in matches:  # Process each match found
                value = match.group(1)  # Extract the numerical value (first capture group)
                unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""  # Extract unit if it exists (second capture group), otherwise empty string
                
                rules.append({  # Add constraint rule dictionary
                    'rule_text': f'Constraint: {constraint_type} = {value} {unit}',  # Format readable rule text like "Constraint: STORAGE_CONSTRAINT = 500 GB"
                    'constraint_type': constraint_type,  # Store the type of constraint (STORAGE, TIME, COMPUTE, etc.)
                    'value': value,  # Store the numerical value separately for easy parsing
                    'unit': unit,  # Store the unit separately (GB, hours, cores, etc.)
                    'rule_type': 'CONSTRAINT',  # Mark as constraint rule
                    'confidence': 0.85  # Assign 85% confidence (high because numerical patterns are specific)
                })
        
        return rules  # Return list of all constraint rules
    
    def extract_dependency_rules(self, content: str) -> List[Dict]:  # Extract dependency and prerequisite information from content
        """Extract dependencies and prerequisites"""  # Docstring explaining this method finds what must be done/installed before something else
        rules = []  # Initialize empty list for dependency rules
        
        dependency_patterns = [  # List of regex patterns that indicate dependencies or prerequisites
            r'depends\s+on\s+(.+)',  # Match "depends on X" pattern
            r'requires?\s+(.+)',  # Match "require X" or "requires X"
            r'needs?\s+(.+)',  # Match "need X" or "needs X"
            r'prerequisite[:\s]+(.+)',  # Match "prerequisite: X" or "prerequisite X"
            r'before\s+(.+)',  # Match "before X" indicating ordering dependency
            r'after\s+(.+)',  # Match "after X" indicating ordering dependency
            r'must\s+have\s+(.+)',  # Match "must have X" as dependency
            r'install\s+(.+)',  # Match "install X" as installation dependency
            r'setup\s+(.+)'  # Match "setup X" as setup dependency
        ]
        
        lines = content.split('\n')  # Split content into lines
        for line in lines:  # Loop through each line
            line_stripped = line.strip()  # Remove whitespace
            if len(line_stripped) < 10:  # Skip very short lines unlikely to contain meaningful dependencies
                continue  # Move to next line
                
            line_lower = line_stripped.lower()  # Convert to lowercase for case-insensitive matching
            for pattern in dependency_patterns:  # Check each dependency pattern
                match = re.search(pattern, line_lower)  # Search for pattern anywhere in line (not just start)
                if match:  # If pattern matched
                    dependency = match.group(1).strip()  # Extract the dependency text (what is required/needed)
                    if 3 < len(dependency) < 100:  # Only add if dependency text is reasonable length (not too short or too long)
                        rules.append({  # Add dependency rule
                            'rule_text': f'Dependency: {dependency}',  # Format as "Dependency: X"
                            'dependency': dependency,  # Store the actual dependency separately
                            'rule_type': 'DEPENDENCY',  # Mark as dependency rule
                            'confidence': 0.72  # Assign 72% confidence (moderate since dependency patterns can be ambiguous)
                        })
                    break  # Stop checking other patterns once we find a match
        
        return rules  # Return list of all dependency rules
    
    def _deduplicate(self, rules: List[Dict]) -> List[Dict]:  # Remove duplicate rules from the list to avoid redundancy
        """Remove duplicate rules"""  # Docstring explaining this private method filters out duplicate entries
        seen = set()  # Initialize empty set to track rule texts we've already encountered (sets provide O(1) lookup)
        unique_rules = []  # Initialize empty list to store deduplicated rules
        
        for rule in rules:  # Loop through each rule dictionary
            rule_text = rule.get('rule_text', str(rule))  # Get the rule_text field, or convert entire rule to string if field doesn't exist
            rule_key = rule_text.lower().strip()[:100]  # Create normalized key: lowercase, strip whitespace, take first 100 chars for comparison
            
            if rule_key not in seen:  # If we haven't seen this rule text before
                seen.add(rule_key)  # Add it to the seen set
                unique_rules.append(rule)  # Add the full rule dictionary to unique list
        
        return unique_rules  # Return deduplicated list of rules
