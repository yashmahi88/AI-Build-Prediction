"""Rule extraction for Yocto builds"""
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class RuleExtractor:
    """Extract rules from content"""
    
    def __init__(self):
        logger.debug("[INIT] RuleExtractor initialized")
    
    def extract_all_rules(self, content: str) -> List[Dict]:
        """Extract all types of rules from content"""
        if not content:
            return []
        
        all_rules = []
        
        # Extract all rule types
        all_rules.extend(self.extract_linguistic_rules(content))
        all_rules.extend(self.extract_structural_rules(content))
        all_rules.extend(self.extract_procedural_rules(content))
        all_rules.extend(self.extract_constraint_rules(content))
        all_rules.extend(self.extract_dependency_rules(content))
        
        # Deduplicate
        unique_rules = self._deduplicate(all_rules)
        
        return unique_rules
    
    def extract_linguistic_rules(self, content: str) -> List[Dict]:
        """Extract rules based on linguistic patterns"""
        rules = []
        lines = content.split('\n')
        
        requirement_patterns = [
            (r'\bmust\b', 'MANDATORY'),
            (r'\bshall\b', 'MANDATORY'), 
            (r'\brequired?\b', 'MANDATORY'),
            (r'\bmandatory\b', 'MANDATORY'),
            (r'\bshould\b', 'RECOMMENDED'),
            (r'\brecommended?\b', 'RECOMMENDED'),
            (r'\bpreferred?\b', 'RECOMMENDED'),
            (r'\bmay\b', 'OPTIONAL'),
            (r'\boptional\b', 'OPTIONAL'),
            (r'\bcould\b', 'OPTIONAL')
        ]
        
        action_patterns = [
            (r'\bensure\b', 'VERIFICATION'),
            (r'\bverify\b', 'VERIFICATION'),
            (r'\bvalidate\b', 'VALIDATION'),
            (r'\bconfigure\b', 'CONFIGURATION'),
            (r'\bsetup\b', 'SETUP'),
            (r'\binstall\b', 'INSTALLATION'),
            (r'\bdeploy\b', 'DEPLOYMENT'),
            (r'\bbuild\b', 'BUILD'),
            (r'\btest\b', 'TESTING'),
            (r'\bmonitor\b', 'MONITORING')
        ]
        
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) < 10 or len(line_stripped) > 200:
                continue
                
            line_lower = line_stripped.lower()
            
            requirement_level = 'UNKNOWN'
            for pattern, level in requirement_patterns:
                if re.search(pattern, line_lower):
                    requirement_level = level
                    break
            
            action_type = 'GENERAL'
            for pattern, action in action_patterns:
                if re.search(pattern, line_lower):
                    action_type = action
                    break
            
            if requirement_level != 'UNKNOWN' or action_type != 'GENERAL':
                rules.append({
                    'rule_text': line_stripped,
                    'requirement_level': requirement_level,
                    'action_type': action_type,
                    'rule_type': 'LINGUISTIC',
                    'confidence': 0.8
                })
        
        return rules
    
    def extract_structural_rules(self, content: str) -> List[Dict]:
        """Extract rules from document structure"""
        rules = []
        lines = content.split('\n')
        
        current_section = ""
        section_patterns = [
            r'^#+\s+(.+)$',
            r'^(.+):$',
            r'^\d+\.\s+(.+)$',
            r'^[A-Z\s]+$'
        ]
        
        requirement_keywords = [
            'requirement', 'prerequisite', 'configuration', 'setup', 'installation',
            'environment', 'dependency', 'constraint', 'rule', 'standard', 'guideline'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match and len(match.group(1)) < 100:
                    current_section = match.group(1).strip()
                    break
            
            if current_section and any(keyword in current_section.lower() for keyword in requirement_keywords):
                if line != current_section and len(line) > 20:
                    rules.append({
                        'rule_text': line,
                        'section': current_section,
                        'rule_type': 'STRUCTURAL',
                        'confidence': 0.75
                    })
        
        return rules
    
    def extract_procedural_rules(self, content: str) -> List[Dict]:
        """Extract procedural rules"""
        rules = []
        
        step_patterns = [
            r'^\d+\.\s+(.+)$',
            r'^step\s+\d+[:\-]\s*(.+)$',
            r'^\w+\)\s+(.+)$',
            r'^•\s+(.+)$',
            r'^-\s+(.+)$',
            r'^\*\s+(.+)$'
        ]
        
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) < 10:
                continue
                
            for pattern in step_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    step_text = match.group(1).strip()
                    if len(step_text) > 15:
                        rules.append({
                            'rule_text': step_text,
                            'rule_type': 'PROCEDURAL',
                            'confidence': 0.65
                        })
                    break
        
        return rules
    
    def extract_constraint_rules(self, content: str) -> List[Dict]:
        """Extract constraints and limits"""
        rules = []
        
        constraint_patterns = [
            (r'(\d+)\s*(GB|MB|TB|gb|mb|tb)', 'STORAGE_CONSTRAINT'),
            (r'(\d+)\s*(hours?|minutes?|seconds?|hrs?|mins?|secs?)', 'TIME_CONSTRAINT'),
            (r'(\d+)\s*(cores?|threads?|CPUs?|processors?)', 'COMPUTE_CONSTRAINT'),
            (r'(\d+)\s*(MB|GB|TB)\s*(RAM|memory)', 'MEMORY_CONSTRAINT'),
            (r'minimum\s+(\d+)', 'MINIMUM_REQUIREMENT'),
            (r'maximum\s+(\d+)', 'MAXIMUM_LIMIT'),
            (r'at\s+least\s+(\d+)', 'MINIMUM_REQUIREMENT'),
            (r'no\s+more\s+than\s+(\d+)', 'MAXIMUM_LIMIT')
        ]
        
        for pattern, constraint_type in constraint_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                
                rules.append({
                    'rule_text': f'Constraint: {constraint_type} = {value} {unit}',
                    'constraint_type': constraint_type,
                    'value': value,
                    'unit': unit,
                    'rule_type': 'CONSTRAINT',
                    'confidence': 0.85
                })
        
        return rules
    
    def extract_dependency_rules(self, content: str) -> List[Dict]:
        """Extract dependencies and prerequisites"""
        rules = []
        
        dependency_patterns = [
            r'depends\s+on\s+(.+)',
            r'requires?\s+(.+)',
            r'needs?\s+(.+)', 
            r'prerequisite[:\s]+(.+)',
            r'before\s+(.+)',
            r'after\s+(.+)',
            r'must\s+have\s+(.+)',
            r'install\s+(.+)',
            r'setup\s+(.+)'
        ]
        
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) < 10:
                continue
                
            line_lower = line_stripped.lower()
            for pattern in dependency_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    dependency = match.group(1).strip()
                    if 3 < len(dependency) < 100:
                        rules.append({
                            'rule_text': f'Dependency: {dependency}',
                            'dependency': dependency,
                            'rule_type': 'DEPENDENCY',
                            'confidence': 0.72
                        })
                    break
        
        return rules
    
    def _deduplicate(self, rules: List[Dict]) -> List[Dict]:
        """Remove duplicate rules"""
        seen = set()
        unique_rules = []
        
        for rule in rules:
            rule_text = rule.get('rule_text', str(rule))
            rule_key = rule_text.lower().strip()[:100]
            
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)
        
        return unique_rules
