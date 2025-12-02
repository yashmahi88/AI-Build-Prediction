"""LLM service for AI-generated suggestions with file-level details"""
import logging
import asyncio
import re
import os
from typing import List, Dict
from langchain_ollama import OllamaLLM
from app.core.prompts import SUGGESTION_PROMPT_TEMPLATE
from app.core.config import get_settings


logger = logging.getLogger(__name__)


class LLMService:
    """Generate AI suggestions for build violations"""
    
    def __init__(self):
        try:
            # Load settings from config
            settings = get_settings()
            
            self.llm = OllamaLLM(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                temperature=settings.ollama_llm_temperature,
                timeout=settings.ollama_llm_timeout
            )
            
            logger.info(
                f"[OK] LLM Service initialized - "
                f"model={settings.ollama_llm_model}, "
                f"temperature={settings.ollama_llm_temperature}, "
                f"timeout={settings.ollama_llm_timeout}s"
            )
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
                logger.info(f"[OK] Generated {len(suggestions)} file-specific suggestions")
            else:
                logger.error("[ERROR] No suggestions generated - parsing failed or LLM gave bad output")
            
            return suggestions
        
        except Exception as e:
            logger.exception(f"[ERROR] Suggestion generation failed: {e}")
            return []
    
    def _extract_file_details(self, violated_rules: List[Dict]) -> Dict:
        """Extract file details from violations"""
        details = {
            'bbappend_files': [],
            'config_files': [],
            'recipe_files': [],
            'environment_issues': [],
            'disk_issues': []
        }
        
        for rule in violated_rules:
            rule_text = rule.get('rule_text', '').lower()
            
            # Extract .bbappend files
            if 'bbappend' in rule_text:
                bbappend_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bbappend)', rule.get('rule_text', ''))
                details['bbappend_files'].extend(bbappend_matches)
            
            # Extract .bb recipe files
            if '.bb' in rule_text:
                bb_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bb)\b', rule.get('rule_text', ''))
                details['recipe_files'].extend(bb_matches)
            
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
        details['recipe_files'] = list(set(details['recipe_files']))
        details['config_files'] = list(set(details['config_files']))
        
        return details
    
    def _scan_workspace(self, workspace_path: str) -> List[str]:
        """Scan workspace for Yocto files"""
        found_files = []
        
        try:
            files_scanned = 0
            max_files = 150
            
            for root, dirs, files in os.walk(workspace_path):
                depth = root.replace(workspace_path, '').count(os.sep)
                if depth > 4:
                    continue
                
                dirs[:] = [d for d in dirs if d not in ['tmp', 'sstate-cache', 'downloads', '.git', 'build', 'cache']]
                
                for file_name in files:
                    if files_scanned >= max_files:
                        break
                    
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        if os.path.getsize(file_path) > 500000:
                            continue
                    except:
                        continue
                    
                    if any(file_name.endswith(ext) for ext in {'.bbappend', '.conf', '.bb', '.inc'}):
                        rel_path = os.path.relpath(file_path, workspace_path)
                        found_files.append(rel_path)
                        files_scanned += 1
            
            logger.info(f"[SCAN] Found {len(found_files)} Yocto files in workspace")
        
        except Exception as e:
            logger.warning(f"[WARN] Workspace scan error: {e}")
        
        return found_files
    
    def _match_files(self, specific_details: Dict, found_files: List[str]) -> Dict:
        """Match found files to violations"""
        matched = {
            'existing_bbappend': {},
            'missing_bbappend': [],
            'existing_config': {},
            'missing_config': [],
            'existing_recipes': {},
            'missing_recipes': []
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
        
        for recipe_file in specific_details['recipe_files']:
            matching = [f for f in found_files if recipe_file in f or os.path.basename(f) == recipe_file]
            if matching:
                matched['existing_recipes'][recipe_file] = matching[0]
            else:
                matched['missing_recipes'].append(recipe_file)
        
        return matched
    
    def _categorize_violations(self, violated_rules: List[Dict]) -> Dict[str, List[str]]:
        """Categorize violations by type to ensure diverse suggestions"""
        categories = {
            'path_directory': [],
            'configuration': [],
            'disk_space': [],
            'recipe_layer': [],
            'environment': [],
            'other': []
        }
        
        for rule in violated_rules:
            rule_text = rule.get('rule_text', '').lower()
            
            if any(kw in rule_text for kw in ['path', 'directory', 'dir', 'mount', 'location']):
                categories['path_directory'].append(rule_text)
            elif any(kw in rule_text for kw in ['disk', 'space', 'storage', 'full', 'capacity']):
                categories['disk_space'].append(rule_text)
            elif any(kw in rule_text for kw in ['.bb', '.bbappend', 'recipe', 'layer', 'meta-']):
                categories['recipe_layer'].append(rule_text)
            elif any(kw in rule_text for kw in ['environment', 'init', 'source', 'export', 'bitbake']):
                categories['environment'].append(rule_text)
            elif any(kw in rule_text for kw in ['conf', 'configuration', 'variable', 'setting']):
                categories['configuration'].append(rule_text)
            else:
                categories['other'].append(rule_text)
        
        return categories
    
    def _generate_with_universal_prompt(self, violated_rules: List[Dict], pipeline_text: str, specific_details: Dict, matched_files: Dict) -> List[str]:
        """Generate with enhanced file-specific context and diversity enforcement"""
        
        # Categorize violations for diversity
        violation_categories = self._categorize_violations(violated_rules)
        
        # Build DETAILED violation context with numbering
        violation_context = ""
        for i, rule in enumerate(violated_rules[:10], 1):  # Show up to 10 violations
            rule_text = rule.get('rule_text', str(rule))
            violation_context += f"{i}. {rule_text}\n"
        
        # Add category summary to help LLM diversify
        violation_context += "\n=== VIOLATION CATEGORIES ===\n"
        for category, violations in violation_categories.items():
            if violations:
                count = len(violations)
                violation_context += f"{category.upper().replace('_', ' ')}: {count} violation(s)\n"
        
        # Build COMPREHENSIVE workspace context
        workspace_context = ""
        
        if matched_files.get('existing_config'):
            workspace_context += "CONFIG FILES FOUND IN WORKSPACE:\n"
            for name, path in matched_files['existing_config'].items():
                workspace_context += f"  ✓ {path} (EXISTS - can be modified)\n"
        else:
            workspace_context += "NO CONFIG FILES FOUND - will need to create conf/local.conf and conf/bblayers.conf\n"
        
        if matched_files.get('existing_bbappend'):
            workspace_context += "\nBBAPPEND FILES FOUND:\n"
            for name, path in matched_files['existing_bbappend'].items():
                workspace_context += f"  ✓ {path} (EXISTS)\n"
        
        if matched_files.get('existing_recipes'):
            workspace_context += "\nRECIPE FILES FOUND:\n"
            for name, path in list(matched_files['existing_recipes'].items())[:5]:
                workspace_context += f"  ✓ {path}\n"
        
        # Extract ACTUAL VALUES from pipeline for context
        pipeline_values = self._extract_pipeline_values(pipeline_text)
        if pipeline_values:
            workspace_context += "\nACTUAL VALUES FROM PIPELINE:\n"
            for key, value in pipeline_values.items():
                workspace_context += f"  {key} = {value}\n"
        
        if matched_files.get('missing_config'):
            workspace_context += "\nMISSING FILES (need to create):\n"
            for name in matched_files['missing_config']:
                workspace_context += f"  ✗ {name} (DOES NOT EXIST)\n"
        
        if matched_files.get('missing_bbappend'):
            workspace_context += "\nMISSING BBAPPEND FILES (may need to create):\n"
            for name in matched_files['missing_bbappend']:
                workspace_context += f"  ✗ {name}\n"
        
        # Use enhanced prompt from prompts.py
        ai_prompt = SUGGESTION_PROMPT_TEMPLATE.format(
            violation_context=violation_context,
            workspace_context=workspace_context,
            pipeline_text=pipeline_text[:800]
        )
        
        try:
            logger.info("[LLM] Calling Ollama with enhanced diversity-focused context...")
            ai_response = self.llm.invoke(ai_prompt)
            
            if not ai_response:
                logger.error("[ERROR] Empty response from LLM")
                return []
            
            logger.info(f"[OK] Got {len(ai_response)} chars from LLM")
            
            # DEBUG: Log the FULL raw response
            logger.info("="*80)
            logger.info("[DEBUG] FULL LLM RAW RESPONSE:")
            logger.info("="*80)
            logger.info(ai_response)
            logger.info("="*80)
            
            # Parse with ENHANCED format (includes WHY field and deduplication)
            suggestions = self._parse_enhanced_format(ai_response)
            
            if suggestions:
                logger.info(f"[OK] Parsed {len(suggestions)} diverse suggestions")
                return suggestions[:5]
            else:
                logger.error("[ERROR] Failed to parse structured suggestions")
                logger.error("[DEBUG] Response did not match expected format with FILE:/CHANGE:/CODE:/WHY:")
                return []
        
        except Exception as e:
            logger.exception(f"[ERROR] LLM call failed: {e}")
            return []
    
    def _extract_pipeline_values(self, pipeline_text: str) -> Dict[str, str]:
        """Extract actual variable values from pipeline code"""
        values = {}
        
        # Extract environment variables
        env_patterns = [
            (r'BASE_PATH\s*=\s*["\']([^"\']+)["\']', 'BASE_PATH'),
            (r'SSTATE_DIR_PATH\s*=\s*["\']([^"\']+)["\']', 'SSTATE_DIR_PATH'),
            (r'DL_DIR_PATH\s*=\s*["\']([^"\']+)["\']', 'DL_DIR_PATH'),
            (r'BUILD_DIR\s*=\s*["\']([^"\']+)["\']', 'BUILD_DIR'),
            (r'YOCTO_WORKSPACE\s*=\s*["\']([^"\']+)["\']', 'YOCTO_WORKSPACE'),
            (r'POKY_DIR\s*=\s*["\']([^"\']+)["\']', 'POKY_DIR'),
            (r'TMPDIR\s*=\s*["\']([^"\']+)["\']', 'TMPDIR'),
        ]
        
        for pattern, var_name in env_patterns:
            match = re.search(pattern, pipeline_text)
            if match:
                values[var_name] = match.group(1)
        
        return values
    
    def _parse_enhanced_format(self, text: str) -> List[str]:
        """Parse FILE:/CHANGE:/CODE:/WHY: format with relaxed validation and debugging"""
        suggestions = []
        seen_codes = set()  # Track unique CODE blocks
        
        logger.info(f"[PARSE] Starting to parse {len(text)} chars")
        
        # Split by bullet points
        blocks = re.split(r'\n•\s*', text)
        logger.info(f"[PARSE] Split into {len(blocks)} blocks")
        
        for idx, block in enumerate(blocks):
            logger.debug(f"[PARSE] Block {idx}: length={len(block)}")
            
            if len(block) < 80:
                logger.debug(f"[PARSE] Block {idx} too short ({len(block)} chars)")
                continue
            
            # Check if block contains required fields
            has_file = 'FILE:' in block.upper()
            has_change = 'CHANGE:' in block.upper()
            has_code = 'CODE:' in block.upper()
            has_why = 'WHY:' in block.upper()
            
            logger.debug(f"[PARSE] Block {idx} fields: FILE={has_file}, CHANGE={has_change}, CODE={has_code}, WHY={has_why}")
            
            # RELAXED: Accept suggestions with at least FILE, CODE, and one other field
            if has_file and has_code and (has_change or has_why):
                suggestion = block.strip()
                
                # Extract CODE field for deduplication
                code_match = re.search(r'CODE:\s*(.+?)(?:\n\s*(?:WHY:|CHANGE:|FILE:|$)|$)', suggestion, re.IGNORECASE | re.DOTALL)
                if code_match:
                    code_content = code_match.group(1).strip()
                    
                    logger.debug(f"[PARSE] Block {idx} CODE content: {code_content[:50]}...")
                    
                    # Create signature for deduplication
                    code_signature = code_content.lower().replace(' ', '').replace('"', '').replace("'", '')
                    
                    # Check for duplicates
                    if code_signature in seen_codes:
                        logger.debug(f"[PARSE] Block {idx} is duplicate")
                        continue
                    
                    # Quality check: ensure CODE has actual content
                    if len(code_content) > 10:  # Lowered from 15
                        # Format nicely
                        suggestion = re.sub(r'FILE:\s*', '\n  FILE: ', suggestion, flags=re.IGNORECASE)
                        suggestion = re.sub(r'CHANGE:\s*', '\n  CHANGE: ', suggestion, flags=re.IGNORECASE)
                        suggestion = re.sub(r'CODE:\s*', '\n  CODE: ', suggestion, flags=re.IGNORECASE)
                        suggestion = re.sub(r'WHY:\s*', '\n  WHY: ', suggestion, flags=re.IGNORECASE)
                        
                        seen_codes.add(code_signature)
                        suggestions.append(suggestion)
                        logger.info(f"[PARSE] ✅ Accepted block {idx}")
                    else:
                        logger.debug(f"[PARSE] Block {idx} CODE too short: {len(code_content)} chars")
                else:
                    logger.debug(f"[PARSE] Block {idx} failed to extract CODE field")
            else:
                missing = []
                if not has_file: missing.append("FILE")
                if not has_change: missing.append("CHANGE")
                if not has_code: missing.append("CODE")
                if not has_why: missing.append("WHY")
                logger.debug(f"[PARSE] Block {idx} missing fields: {', '.join(missing)}")
        
        if not suggestions:
            logger.error("[PARSE] No valid suggestions found after parsing all blocks")
        else:
            logger.info(f"[PARSE] Successfully parsed {len(suggestions)} suggestions")
        
        return suggestions
