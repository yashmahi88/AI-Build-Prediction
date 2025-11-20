"""LLM service for AI-generated suggestions with file-level details"""
import logging
import asyncio
import re
import os
from typing import List, Dict
from langchain_ollama import OllamaLLM


logger = logging.getLogger(__name__)
suggestion_cache = {}


class LLMService:
    """Generate AI suggestions for build violations"""
    
    def __init__(self):
        try:
            self.llm = OllamaLLM(
                model= "qwen2.5:1.5b", ##"mistral:7b-instruct-q4_0", ##"codellama:7b", ##"codegemma:2b", ##"qwen3-coder:480b-cloud",
                base_url="http://localhost:11434",
                temperature=0.1,
                timeout=180
            )
            logger.info("[OK] LLM Service initialized")
        except Exception as e:
            logger.error(f"[ERROR] LLM init failed: {e}")
            self.llm = None
    
    async def generate_suggestions(self, violated_rules: List[Dict], pipeline_text: str, workspace_path: str = None) -> List[str]:
        """Generate AI suggestions WITH file-specific details"""
        
        if not self.llm:
            logger.warning("[WARN] LLM not initialized")
            return []
        
        if not violated_rules:
            logger.info("[OK] No violations")
            return []
        
        logger.info(f"[LLM] Generating file-specific suggestions...")
        
        try:
            # Extract file details from violations
            specific_details = self._extract_file_details(violated_rules)
            logger.info(f"[LLM] Extracted files: {specific_details}")
            
            # Scan workspace for actual files
            matched_files = {}
            if workspace_path and os.path.exists(workspace_path):
                found_files = self._scan_workspace(workspace_path)
                matched_files = self._match_files(specific_details, found_files)
                logger.info(f"[LLM] Matched files: {matched_files}")
            
            # Run LLM with comprehensive context
            suggestions = await asyncio.to_thread(
                self._generate_with_universal_prompt,
                violated_rules,
                pipeline_text,
                specific_details,
                matched_files
            )
            
            if suggestions:
                logger.info(f"[OK] Generated {len(suggestions)} suggestions")
            else:
                logger.warning("[WARN] No suggestions from LLM")
            
            return suggestions
        
        except Exception as e:
            logger.error(f"[ERROR] {e}", exc_info=True)
            return self._generate_fallback_with_files(violated_rules)
    
    def _extract_file_details(self, violated_rules: List[Dict]) -> Dict:
        """Extract file details from violations - FROM ORIGINAL CODE"""
        details = {
            'bbappend_files': [],
            'config_files': [],
            'environment_issues': [],
            'disk_issues': []
        }
        
        for rule in violated_rules:
            rule_text = rule.get('rule_text', '').lower()
            
            # Extract .bbappend files
            if 'bbappend' in rule_text:
                bbappend_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bbappend)', rule.get('rule_text', ''))
                details['bbappend_files'].extend(bbappend_matches)
            
            # Environment issues
            if 'environment' in rule_text:
                details['environment_issues'].append('BitBake environment initialization')
            
            # Disk space issues
            if 'disk space' in rule_text or 'storage' in rule_text:
                details['disk_issues'].append('Insufficient disk space')
            
            # Config files
            if 'local.conf' in rule_text:
                details['config_files'].append('local.conf')
            if 'bblayers.conf' in rule_text:
                details['config_files'].append('bblayers.conf')
        
        details['bbappend_files'] = list(set(details['bbappend_files']))
        details['config_files'] = list(set(details['config_files']))
        
        return details
    
    def _scan_workspace(self, workspace_path: str) -> List[str]:
        """Scan workspace for files"""
        found_files = []
        
        try:
            files_scanned = 0
            max_files = 100
            
            for root, dirs, files in os.walk(workspace_path):
                depth = root.replace(workspace_path, '').count(os.sep)
                if depth > 3:
                    continue
                
                dirs[:] = [d for d in dirs if d not in ['tmp', 'sstate-cache', 'downloads', '.git', 'build']]
                
                for file_name in files:
                    if files_scanned >= max_files:
                        break
                    
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        if os.path.getsize(file_path) > 500000:
                            continue
                    except:
                        continue
                    
                    if any(file_name.endswith(ext) for ext in {'.bbappend', '.conf', '.bb'}):
                        rel_path = os.path.relpath(file_path, workspace_path)
                        found_files.append(rel_path)
                        files_scanned += 1
        
        except Exception as e:
            logger.warning(f"[WARN] Workspace scan error: {e}")
        
        return found_files
    
    def _match_files(self, specific_details: Dict, found_files: List[str]) -> Dict:
        """Match found files to violations"""
        matched = {
            'existing_bbappend': {},
            'missing_bbappend': [],
            'existing_config': {},
            'missing_config': []
        }
        
        for bbappend_file in specific_details['bbappend_files']:
            matching = [f for f in found_files if bbappend_file in f or os.path.basename(f) == bbappend_file]
            if matching:
                matched['existing_bbappend'][bbappend_file] = matching[0]
            else:
                matched['missing_bbappend'].append(bbappend_file)
        
        for config_file in specific_details['config_files']:
            matching = [f for f in found_files if config_file in f]
            if matching:
                matched['existing_config'][config_file] = matching[0]
            else:
                matched['missing_config'].append(config_file)
        
        return matched
    
    def _generate_with_universal_prompt(self, violated_rules: List[Dict], pipeline_text: str, specific_details: Dict, matched_files: Dict) -> List[str]:
        """Generate with comprehensive context - FROM ORIGINAL CODE LOGIC"""
        
        violation_context = "\n".join([
            f"• {rule.get('rule_text', str(rule))}"
            for rule in violated_rules[:7]
        ])
        
        workspace_context = "WORKSPACE FILES:\n"
        if matched_files.get('existing_bbappend'):
            workspace_context += "EXISTING .bbappend files:\n"
            for name, path in matched_files['existing_bbappend'].items():
                workspace_context += f"  - {path}\n"
        
        if matched_files.get('missing_bbappend'):
            workspace_context += "MISSING .bbappend files:\n"
            for name in matched_files['missing_bbappend']:
                workspace_context += f"  - {name}\n"
        
        if matched_files.get('existing_config'):
            workspace_context += "EXISTING config files:\n"
            for name, path in matched_files['existing_config'].items():
                workspace_context += f"  - {path}\n"
        
        if matched_files.get('missing_config'):
            workspace_context += "MISSING config files:\n"
            for name in matched_files['missing_config']:
                workspace_context += f"  - {name}\n"
        
        ai_prompt = f"""Yocto build expert: Fix these violations with SPECIFIC file changes:

VIOLATIONS:
{violation_context}

{workspace_context}

PIPELINE:
{pipeline_text[:600]}

Generate 5 COMPLETE solutions using bullet points (•). Each must have:
• Issue description
  FILE: exact/file/path/to/modify
  CHANGE: what to add/modify
  CODE: exact command or configuration
  (+X% confidence)

Be specific. Include real file paths. Provide complete code. Make it actionable."""

        try:
            logger.info("[LLM] Calling Ollama...")
            ai_response = self.llm.invoke(ai_prompt)
            
            if not ai_response:
                logger.warning("[WARN] Empty response")
                return []
            
            logger.info(f"[OK] Got {len(ai_response)} chars")
            logger.debug(f"[PREVIEW] {ai_response[:200]}...")
            
            # Parse using ORIGINAL logic
            suggestions = self._parse_universal_response(ai_response)
            logger.info(f"[OK] Parsed {len(suggestions)} suggestions")
            
            return suggestions[:5]
        
        except Exception as e:
            logger.exception(f"[ERROR] LLM call failed: {e}")
            return []
    
    def _parse_universal_response(self, ai_response: str) -> List[str]:
        """Parse using ORIGINAL multi-strategy approach"""
        suggestions = []
        
        # Strategy 1: Bullet format
        suggestions = self._parse_bullet_format(ai_response)
        if suggestions:
            return suggestions
        
        # Strategy 2: Numbered format
        suggestions = self._parse_numbered_format(ai_response)
        if suggestions:
            return suggestions
        
        # Strategy 3: Confidence indicators
        suggestions = self._parse_confidence_format(ai_response)
        if suggestions:
            return suggestions
        
        # Strategy 4: Split by file mentions
        suggestions = self._parse_file_mentions(ai_response)
        return suggestions
    
    def _parse_bullet_format(self, text: str) -> List[str]:
        """Parse bullet points"""
        suggestions = []
        lines = text.split('\n')
        current = ""
        
        for line in lines:
            if line.strip().startswith('•'):
                if current and len(current) > 40:
                    suggestions.append(current.strip())
                current = line.strip()[1:].strip()
            elif current:
                current += '\n' + line
        
        if current and len(current) > 40:
            suggestions.append(current.strip())
        
        return suggestions
    
    def _parse_numbered_format(self, text: str) -> List[str]:
        """Parse numbered items"""
        suggestions = []
        lines = text.split('\n')
        current = ""
        
        for line in lines:
            if re.match(r'^\d+[\.\)]\s', line.strip()):
                if current and len(current) > 40:
                    suggestions.append(current.strip())
                current = re.sub(r'^\d+[\.\)]\s+', '', line.strip())
            elif current:
                current += '\n' + line
        
        if current and len(current) > 40:
            suggestions.append(current.strip())
        
        return suggestions
    
    def _parse_confidence_format(self, text: str) -> List[str]:
        """Parse by confidence indicators"""
        suggestions = []
        pattern = r'\(\+\d+%\s*confidence\)'
        parts = re.split(pattern, text)
        
        for i in range(len(parts) - 1):
            part = parts[i].strip()
            if len(part) > 40:
                conf_match = re.search(pattern, text[text.find(part) + len(part):])
                if conf_match:
                    suggestions.append(part + ' ' + conf_match.group())
        
        return suggestions
    
    def _parse_file_mentions(self, text: str) -> List[str]:
        """Parse by FILE: mentions"""
        suggestions = []
        
        # Look for FILE: patterns
        file_pattern = r'FILE:.*?(?=FILE:|$)'
        matches = re.finditer(file_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            chunk = match.group().strip()
            if len(chunk) > 40:
                suggestions.append(chunk)
        
        return suggestions
    
    def _generate_fallback_with_files(self, violated_rules: List[Dict]) -> List[str]:
        """Generate fallback suggestions with file info"""
        suggestions = []
        
        for i, rule in enumerate(violated_rules[:5], 1):
            text = rule.get('rule_text', str(rule))[:80]
            
            suggestion = f"""• Fix violation {i}: {text}
  FILE: conf/local.conf
  CHANGE: Add appropriate configuration
  CODE: # Update local.conf with needed settings
  """
            
            suggestions.append(suggestion)
        
        logger.info(f"[FALLBACK] Generated {len(suggestions)} suggestions")
        return suggestions
