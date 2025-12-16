<<<<<<< Updated upstream
"""LLM service for AI-generated suggestions with file-level details"""
import logging
import asyncio
import re
import os
from typing import List, Dict
from langchain_ollama import OllamaLLM


logger = logging.getLogger(__name__)
suggestion_cache = {}
=======
"""LLM service for AI-generated suggestions with file-level details"""  # Module docstring describing this file uses Ollama LLM to generate actionable build fix suggestions
import logging  # Standard Python logging library for tracking LLM operations
import asyncio  # Python's async library for non-blocking LLM calls
import re  # Regular expression library for parsing LLM responses and extracting file paths
import os  # Operating system interface for file path operations and workspace scanning
from typing import List, Dict  # Type hints for function signatures: List for arrays, Dict for dictionaries
from langchain_ollama import OllamaLLM  # LangChain wrapper for Ollama local LLM (provides standardized interface)
from app.core.prompts import SUGGESTION_PROMPT_TEMPLATE  # Import prompt template for LLM (defines structure for suggestions)
from app.core.config import get_settings  # Function to load application configuration (Ollama URL, model name, etc.)



logger = logging.getLogger(__name__)  # Create logger instance for this module to output LLM-related logs
>>>>>>> Stashed changes



class LLMService:  # Service class that generates AI-powered suggestions for fixing build violations
    """Generate AI suggestions for build violations"""  # Docstring explaining this class uses LLM to create actionable fixes
    
<<<<<<< Updated upstream
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
=======
    def __init__(self):  # Constructor that initializes connection to Ollama LLM
        try:  # Wrap initialization in try-except to handle LLM connection errors
            # Load settings from config
            settings = get_settings()  # Load configuration (model name, base URL, temperature, timeout)
            
            self.llm = OllamaLLM(  # Create Ollama LLM instance using LangChain wrapper
                model=settings.ollama_llm_model,  # Model name from config (e.g., "llama2", "mistral", "codellama")
                base_url=settings.ollama_base_url,  # Ollama server URL from config (e.g., "http://localhost:11434")
                temperature=settings.ollama_llm_temperature,  # Temperature controls randomness (0.0=deterministic, 1.0=creative) from config
                timeout=settings.ollama_llm_timeout  # Timeout in seconds for LLM calls from config (prevents hanging)
            )
            
            logger.info(  # Log successful initialization with all configuration details
                f"[OK] LLM Service initialized - "
                f"model={settings.ollama_llm_model}, "  # Show which model is loaded
                f"temperature={settings.ollama_llm_temperature}, "  # Show creativity setting
                f"timeout={settings.ollama_llm_timeout}s"  # Show timeout value
            )
        except Exception as e:  # Catch any initialization errors (LLM not running, wrong URL, etc.)
            logger.error(f"[ERROR] LLM init failed: {e}")  # Log error details
            self.llm = None  # Set llm to None so other methods know LLM is unavailable
>>>>>>> Stashed changes
    
    async def generate_suggestions(self, violated_rules: List[Dict], pipeline_text: str, workspace_path: str = None) -> List[str]:  # Async method that generates file-specific suggestions for violated rules
        """Generate AI suggestions WITH file-specific details"""  # Docstring explaining this creates suggestions with actual file paths from workspace
        
        if not self.llm:  # Check if LLM failed to initialize
            logger.warning("[WARN] LLM not initialized")  # Log warning that LLM is unavailable
            return []  # Return empty list since we can't generate suggestions without LLM
        
        if not violated_rules:  # Check if there are no violations to address
            logger.info("[OK] No violations")  # Log that no suggestions are needed
            return []  # Return empty list since nothing to fix
        
        logger.info(f"[LLM] Generating file-specific suggestions...")  # Log start of suggestion generation
        
        try:  # Wrap suggestion generation in try-except to handle errors gracefully
            # Extract file details from violations
            specific_details = self._extract_file_details(violated_rules)  # Parse violation text to extract mentioned files (.bbappend, .conf, .bb, etc.)
            logger.info(f"[LLM] Extracted files: {specific_details}")  # Log what file references were found in violations
            
            # Scan workspace for actual files
            matched_files = {}  # Initialize empty dict to store matched files (maps logical names to actual paths)
            if workspace_path and os.path.exists(workspace_path):  # Only scan if workspace path was provided and exists
                found_files = self._scan_workspace(workspace_path)  # Walk workspace directory tree to find all Yocto files (.bb, .bbappend, .conf)
                matched_files = self._match_files(specific_details, found_files)  # Match extracted file references to actual files found in workspace
                logger.info(f"[LLM] Matched files: {matched_files}")  # Log which files were successfully matched
            
            # Run LLM with comprehensive context
            suggestions = await asyncio.to_thread(  # Run blocking LLM call in thread pool (prevents blocking async event loop)
                self._generate_with_universal_prompt,  # Method that builds prompt and calls LLM
                violated_rules,  # Violated rules to address
                pipeline_text,  # Pipeline content for context
                specific_details,  # Extracted file references
                matched_files  # Actual file paths from workspace
            )
            
<<<<<<< Updated upstream
            if suggestions:
                logger.info(f"[OK] Generated {len(suggestions)} suggestions")
            else:
                logger.warning("[WARN] No suggestions from LLM")
=======
            if suggestions:  # Check if LLM generated any suggestions
                logger.info(f"[OK] Generated {len(suggestions)} file-specific suggestions")  # Log success with count
            else:  # No suggestions were generated
                logger.error("[ERROR] No suggestions generated - parsing failed or LLM gave bad output")  # Log error (either LLM failed or response parsing failed)
>>>>>>> Stashed changes
            
            return suggestions  # Return list of suggestions (may be empty)
        
<<<<<<< Updated upstream
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
=======
        except Exception as e:  # Catch any errors during suggestion generation
            logger.exception(f"[ERROR] Suggestion generation failed: {e}")  # Log error with full traceback
            return []  # Return empty list on error
    
    def _extract_file_details(self, violated_rules: List[Dict]) -> Dict:  # Method to parse violation text and extract file references
        """Extract file details from violations"""  # Docstring explaining this finds .bbappend, .bb, .conf files mentioned in rules
        details = {  # Initialize dict with empty lists for each file type category
            'bbappend_files': [],  # List to store .bbappend file references
            'config_files': [],  # List to store .conf file references (local.conf, bblayers.conf)
            'recipe_files': [],  # List to store .bb recipe file references
            'environment_issues': [],  # List to store environment-related issues
            'disk_issues': []  # List to store disk space-related issues
>>>>>>> Stashed changes
        }
        
        for rule in violated_rules:  # Loop through each violated rule
            rule_text = rule.get('rule_text', '').lower()  # Extract rule text and convert to lowercase for case-insensitive matching
            
            # Extract .bbappend files
            if 'bbappend' in rule_text:  # Check if rule mentions bbappend files
                bbappend_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bbappend)', rule.get('rule_text', ''))  # Regex to find .bbappend filenames (handles versions like recipe_1.0.bbappend)
                details['bbappend_files'].extend(bbappend_matches)  # Add all found bbappend files to list
            
<<<<<<< Updated upstream
=======
            # Extract .bb recipe files
            if '.bb' in rule_text:  # Check if rule mentions .bb files
                bb_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bb)\b', rule.get('rule_text', ''))  # Regex to find .bb filenames (word boundary \b prevents matching .bbappend)
                details['recipe_files'].extend(bb_matches)  # Add all found recipe files to list
            
>>>>>>> Stashed changes
            # Environment issues
            if 'environment' in rule_text:  # Check if rule mentions environment setup
                details['environment_issues'].append('BitBake environment initialization')  # Add generic environment issue marker
            
            # Disk space issues
            if 'disk space' in rule_text or 'storage' in rule_text:  # Check if rule mentions disk/storage problems
                details['disk_issues'].append('Insufficient disk space')  # Add disk space issue marker
            
            # Config files
            if 'local.conf' in rule_text:  # Check if rule mentions local.conf
                details['config_files'].append('local.conf')  # Add local.conf to config files list
            if 'bblayers.conf' in rule_text:  # Check if rule mentions bblayers.conf
                details['config_files'].append('bblayers.conf')  # Add bblayers.conf to config files list
        
<<<<<<< Updated upstream
        details['bbappend_files'] = list(set(details['bbappend_files']))
        details['config_files'] = list(set(details['config_files']))
=======
        details['bbappend_files'] = list(set(details['bbappend_files']))  # Remove duplicates by converting to set then back to list
        details['recipe_files'] = list(set(details['recipe_files']))  # Remove duplicates from recipe files
        details['config_files'] = list(set(details['config_files']))  # Remove duplicates from config files
>>>>>>> Stashed changes
        
        return details  # Return dict with all extracted file references
    
<<<<<<< Updated upstream
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
=======
    def _scan_workspace(self, workspace_path: str) -> List[str]:  # Method to recursively scan workspace for Yocto-related files
        """Scan workspace for Yocto files"""  # Docstring explaining this walks directory tree to find .bb, .bbappend, .conf files
        found_files = []  # Initialize empty list to store relative paths of found files
        
        try:  # Wrap scanning in try-except to handle filesystem errors
            files_scanned = 0  # Counter to track how many files we've processed
            max_files = 150  # Limit to prevent scanning too many files (performance optimization)
            
            for root, dirs, files in os.walk(workspace_path):  # Walk directory tree recursively (root=current dir, dirs=subdirs, files=files in current dir)
                depth = root.replace(workspace_path, '').count(os.sep)  # Calculate directory depth by counting path separators
                if depth > 4:  # Limit recursion depth to 4 levels (prevents scanning too deep)
                    continue  # Skip this directory and its subdirectories
                
                dirs[:] = [d for d in dirs if d not in ['tmp', 'sstate-cache', 'downloads', '.git', 'build', 'cache']]  # Filter out directories we don't want to scan (build artifacts, cache, git)
>>>>>>> Stashed changes
                
                for file_name in files:  # Loop through files in current directory
                    if files_scanned >= max_files:  # Check if we've hit the file limit
                        break  # Stop scanning to avoid performance issues
                    
                    file_path = os.path.join(root, file_name)  # Build full file path by joining directory and filename
                    
                    try:  # Wrap file size check in try-except
                        if os.path.getsize(file_path) > 500000:  # Check if file is larger than 500KB (500,000 bytes)
                            continue  # Skip large files to avoid reading huge files into memory
                    except:  # Catch any errors getting file size (permission denied, etc.)
                        continue  # Skip this file on error
                    
<<<<<<< Updated upstream
                    if any(file_name.endswith(ext) for ext in {'.bbappend', '.conf', '.bb'}):
                        rel_path = os.path.relpath(file_path, workspace_path)
                        found_files.append(rel_path)
                        files_scanned += 1
=======
                    if any(file_name.endswith(ext) for ext in {'.bbappend', '.conf', '.bb', '.inc'}):  # Check if file has Yocto-related extension (.bbappend, .conf, .bb, .inc)
                        rel_path = os.path.relpath(file_path, workspace_path)  # Convert absolute path to relative path (relative to workspace root)
                        found_files.append(rel_path)  # Add relative path to found files list
                        files_scanned += 1  # Increment counter
            
            logger.info(f"[SCAN] Found {len(found_files)} Yocto files in workspace")  # Log total number of files found
>>>>>>> Stashed changes
        
        except Exception as e:  # Catch any errors during workspace scanning
            logger.warning(f"[WARN] Workspace scan error: {e}")  # Log warning but don't fail (return whatever files we found)
        
        return found_files  # Return list of relative paths to Yocto files
    
<<<<<<< Updated upstream
    def _match_files(self, specific_details: Dict, found_files: List[str]) -> Dict:
        """Match found files to violations"""
        matched = {
            'existing_bbappend': {},
            'missing_bbappend': [],
            'existing_config': {},
            'missing_config': []
=======
    def _match_files(self, specific_details: Dict, found_files: List[str]) -> Dict:  # Method to match extracted file references to actual files found in workspace
        """Match found files to violations"""  # Docstring explaining this maps logical file references to actual workspace paths
        matched = {  # Initialize dict to categorize files as existing or missing
            'existing_bbappend': {},  # Dict mapping bbappend file references to actual paths (for files that exist)
            'missing_bbappend': [],  # List of bbappend files mentioned in violations but not found in workspace
            'existing_config': {},  # Dict mapping config file references to actual paths
            'missing_config': [],  # List of config files mentioned but not found
            'existing_recipes': {},  # Dict mapping recipe file references to actual paths
            'missing_recipes': []  # List of recipe files mentioned but not found
>>>>>>> Stashed changes
        }
        
        for bbappend_file in specific_details['bbappend_files']:  # Loop through each bbappend file extracted from violations
            matching = [f for f in found_files if bbappend_file in f or os.path.basename(f) == bbappend_file]  # Find files where bbappend name appears in path or matches basename exactly
            if matching:  # If we found at least one matching file
                matched['existing_bbappend'][bbappend_file] = matching[0]  # Store first match (reference -> actual path)
            else:  # No matching file found
                matched['missing_bbappend'].append(bbappend_file)  # Add to missing list (LLM can suggest creating it)
        
        for config_file in specific_details['config_files']:  # Loop through each config file extracted from violations
            matching = [f for f in found_files if config_file in f]  # Find files where config name appears in path
            if matching:  # If we found at least one matching file
                matched['existing_config'][config_file] = matching[0]  # Store first match
            else:  # No matching file found
                matched['missing_config'].append(config_file)  # Add to missing list
        
<<<<<<< Updated upstream
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
=======
        for recipe_file in specific_details['recipe_files']:  # Loop through each recipe file extracted from violations
            matching = [f for f in found_files if recipe_file in f or os.path.basename(f) == recipe_file]  # Find files where recipe name appears in path or matches basename
            if matching:  # If we found at least one matching file
                matched['existing_recipes'][recipe_file] = matching[0]  # Store first match
            else:  # No matching file found
                matched['missing_recipes'].append(recipe_file)  # Add to missing list
        
        return matched  # Return dict with categorized file matches
    
    def _categorize_violations(self, violated_rules: List[Dict]) -> Dict[str, List[str]]:  # Method to group violations by type to ensure diverse suggestions
        """Categorize violations by type to ensure diverse suggestions"""  # Docstring explaining this groups violations into categories for better LLM prompting
        categories = {  # Initialize dict with empty lists for each violation category
            'path_directory': [],  # Path/directory-related violations
            'configuration': [],  # Configuration variable violations
            'disk_space': [],  # Disk space/storage violations
            'recipe_layer': [],  # Recipe/layer-related violations
            'environment': [],  # Environment setup violations
            'other': []  # Uncategorized violations
        }
        
        for rule in violated_rules:  # Loop through each violated rule
            rule_text = rule.get('rule_text', '').lower()  # Extract rule text in lowercase for keyword matching
            
            if any(kw in rule_text for kw in ['path', 'directory', 'dir', 'mount', 'location']):  # Check if rule mentions path/directory keywords
                categories['path_directory'].append(rule_text)  # Add to path_directory category
            elif any(kw in rule_text for kw in ['disk', 'space', 'storage', 'full', 'capacity']):  # Check if rule mentions disk space keywords
                categories['disk_space'].append(rule_text)  # Add to disk_space category
            elif any(kw in rule_text for kw in ['.bb', '.bbappend', 'recipe', 'layer', 'meta-']):  # Check if rule mentions recipe/layer keywords
                categories['recipe_layer'].append(rule_text)  # Add to recipe_layer category
            elif any(kw in rule_text for kw in ['environment', 'init', 'source', 'export', 'bitbake']):  # Check if rule mentions environment keywords
                categories['environment'].append(rule_text)  # Add to environment category
            elif any(kw in rule_text for kw in ['conf', 'configuration', 'variable', 'setting']):  # Check if rule mentions configuration keywords
                categories['configuration'].append(rule_text)  # Add to configuration category
            else:  # No category keywords matched
                categories['other'].append(rule_text)  # Add to other category (catchall)
        
        return categories  # Return dict with categorized violations
    
    def _generate_with_universal_prompt(self, violated_rules: List[Dict], pipeline_text: str, specific_details: Dict, matched_files: Dict) -> List[str]:  # Method that builds comprehensive prompt and calls LLM to generate suggestions
        """Generate with enhanced file-specific context and diversity enforcement"""  # Docstring explaining this creates detailed prompt with workspace context
        
        # Categorize violations for diversity
        violation_categories = self._categorize_violations(violated_rules)  # Group violations by type to help LLM generate diverse suggestions
        
        # Build DETAILED violation context with numbering
        violation_context = ""  # Initialize empty string for violation section of prompt
        for i, rule in enumerate(violated_rules[:10], 1):  # Loop through first 10 violations (enumerate starts at 1 for human-readable numbering)
            rule_text = rule.get('rule_text', str(rule))  # Extract rule text (handle dict/string formats)
            violation_context += f"{i}. {rule_text}\n"  # Add numbered violation to context
        
        # Add category summary to help LLM diversify
        violation_context += "\n=== VIOLATION CATEGORIES ===\n"  # Add section header
        for category, violations in violation_categories.items():  # Loop through each category
            if violations:  # Only show categories that have violations
                count = len(violations)  # Count violations in this category
                violation_context += f"{category.upper().replace('_', ' ')}: {count} violation(s)\n"  # Add category summary line (replace underscores with spaces for readability)
        
        # Build COMPREHENSIVE workspace context
        workspace_context = ""  # Initialize empty string for workspace section of prompt
        
        if matched_files.get('existing_config'):  # Check if we found any config files in workspace
            workspace_context += "CONFIG FILES FOUND IN WORKSPACE:\n"  # Add section header
            for name, path in matched_files['existing_config'].items():  # Loop through each found config file
                workspace_context += f"  ✓ {path} (EXISTS - can be modified)\n"  # Show file path with checkmark (indicate it exists and can be edited)
        else:  # No config files found
            workspace_context += "NO CONFIG FILES FOUND - will need to create conf/local.conf and conf/bblayers.conf\n"  # Tell LLM that config files need to be created
        
        if matched_files.get('existing_bbappend'):  # Check if we found any bbappend files
            workspace_context += "\nBBAPPEND FILES FOUND:\n"  # Add section header
            for name, path in matched_files['existing_bbappend'].items():  # Loop through each found bbappend file
                workspace_context += f"  ✓ {path} (EXISTS)\n"  # Show file path with checkmark
        
        if matched_files.get('existing_recipes'):  # Check if we found any recipe files
            workspace_context += "\nRECIPE FILES FOUND:\n"  # Add section header
            for name, path in list(matched_files['existing_recipes'].items())[:5]:  # Loop through first 5 found recipes ([:5] limits output)
                workspace_context += f"  ✓ {path}\n"  # Show file path with checkmark
        
        # Extract ACTUAL VALUES from pipeline for context
        pipeline_values = self._extract_pipeline_values(pipeline_text)  # Parse pipeline content to extract variable assignments (BASE_PATH, SSTATE_DIR, etc.)
        if pipeline_values:  # If we extracted any values
            workspace_context += "\nACTUAL VALUES FROM PIPELINE:\n"  # Add section header
            for key, value in pipeline_values.items():  # Loop through each extracted variable
                workspace_context += f"  {key} = {value}\n"  # Show variable assignment (helps LLM understand current configuration)
        
        if matched_files.get('missing_config'):  # Check if any config files were mentioned but not found
            workspace_context += "\nMISSING FILES (need to create):\n"  # Add section header
            for name in matched_files['missing_config']:  # Loop through missing config files
                workspace_context += f"  ✗ {name} (DOES NOT EXIST)\n"  # Show file with X mark (indicate it needs to be created)
        
        if matched_files.get('missing_bbappend'):  # Check if any bbappend files were mentioned but not found
            workspace_context += "\nMISSING BBAPPEND FILES (may need to create):\n"  # Add section header
            for name in matched_files['missing_bbappend']:  # Loop through missing bbappend files
                workspace_context += f"  ✗ {name}\n"  # Show file with X mark
        
        # Use enhanced prompt from prompts.py
        ai_prompt = SUGGESTION_PROMPT_TEMPLATE.format(  # Build final prompt by filling in template placeholders
            violation_context=violation_context,  # Insert violation details section
            workspace_context=workspace_context,  # Insert workspace files section
            pipeline_text=pipeline_text[:800]  # Insert truncated pipeline content ([:800] limits to 800 chars to avoid overwhelming LLM context)
        )
        
        try:  # Wrap LLM call in try-except to handle errors
            logger.info("[LLM] Calling Ollama with enhanced diversity-focused context...")  # Log that we're making LLM call
            ai_response = self.llm.invoke(ai_prompt)  # Call LLM with constructed prompt (blocking call, but we're in thread pool)
            
            if not ai_response:  # Check if LLM returned empty response
                logger.error("[ERROR] Empty response from LLM")  # Log error
                return []  # Return empty list
            
            logger.info(f"[OK] Got {len(ai_response)} chars from LLM")  # Log response length
            
            # DEBUG: Log the FULL raw response
            logger.info("="*80)  # Log separator line (80 equals signs)
            logger.info("[DEBUG] FULL LLM RAW RESPONSE:")  # Log debug header
            logger.info("="*80)  # Log separator line
            logger.info(ai_response)  # Log entire raw LLM response for debugging
            logger.info("="*80)  # Log separator line
            
            # Parse with ENHANCED format (includes WHY field and deduplication)
            suggestions = self._parse_enhanced_format(ai_response)  # Parse LLM response to extract structured suggestions (FILE:/CHANGE:/CODE:/WHY: format)
            
            if suggestions:  # Check if parsing succeeded
                logger.info(f"[OK] Parsed {len(suggestions)} diverse suggestions")  # Log number of successfully parsed suggestions
                return suggestions[:5]  # Return first 5 suggestions ([:5] limits output)
            else:  # Parsing failed
                logger.error("[ERROR] Failed to parse structured suggestions")  # Log parsing error
                logger.error("[DEBUG] Response did not match expected format with FILE:/CHANGE:/CODE:/WHY:")  # Log what went wrong
                return []  # Return empty list
>>>>>>> Stashed changes
        
        except Exception as e:  # Catch any errors during LLM call or parsing
            logger.exception(f"[ERROR] LLM call failed: {e}")  # Log error with full traceback
            return []  # Return empty list on error
    
<<<<<<< Updated upstream
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
=======
    def _extract_pipeline_values(self, pipeline_text: str) -> Dict[str, str]:  # Method to parse pipeline content and extract variable assignments
        """Extract actual variable values from pipeline code"""  # Docstring explaining this finds environment variables and their values
        values = {}  # Initialize empty dict to store variable assignments
        
        # Extract environment variables
        env_patterns = [  # List of tuples with regex patterns for common Yocto variables and their names
            (r'BASE_PATH\s*=\s*["\']([^"\']+)["\']', 'BASE_PATH'),  # Match BASE_PATH = "value" assignment
            (r'SSTATE_DIR_PATH\s*=\s*["\']([^"\']+)["\']', 'SSTATE_DIR_PATH'),  # Match SSTATE_DIR_PATH = "value"
            (r'DL_DIR_PATH\s*=\s*["\']([^"\']+)["\']', 'DL_DIR_PATH'),  # Match DL_DIR_PATH = "value"
            (r'BUILD_DIR\s*=\s*["\']([^"\']+)["\']', 'BUILD_DIR'),  # Match BUILD_DIR = "value"
            (r'YOCTO_WORKSPACE\s*=\s*["\']([^"\']+)["\']', 'YOCTO_WORKSPACE'),  # Match YOCTO_WORKSPACE = "value"
            (r'POKY_DIR\s*=\s*["\']([^"\']+)["\']', 'POKY_DIR'),  # Match POKY_DIR = "value"
            (r'TMPDIR\s*=\s*["\']([^"\']+)["\']', 'TMPDIR'),  # Match TMPDIR = "value"
        ]
        
        for pattern, var_name in env_patterns:  # Loop through each pattern tuple
            match = re.search(pattern, pipeline_text)  # Search for pattern in pipeline text
            if match:  # If pattern matched
                values[var_name] = match.group(1)  # Extract captured value (first capture group) and store in dict
        
        return values  # Return dict with all extracted variable assignments
    
    def _parse_enhanced_format(self, text: str) -> List[str]:  # Method to parse LLM response into structured suggestions
        """Parse FILE:/CHANGE:/CODE:/WHY: format with relaxed validation and debugging"""  # Docstring explaining this extracts suggestions in expected format with deduplication
        suggestions = []  # Initialize empty list to store parsed suggestions
        seen_codes = set()  # Initialize set to track unique CODE blocks (for deduplication)
        
        logger.info(f"[PARSE] Starting to parse {len(text)} chars")  # Log start of parsing with response length
        
        # Split by bullet points
        blocks = re.split(r'\n•\s*', text)  # Split response by bullet points (• character) to separate suggestions
        logger.info(f"[PARSE] Split into {len(blocks)} blocks")  # Log number of blocks found
        
        for idx, block in enumerate(blocks):  # Loop through each block with index
            logger.debug(f"[PARSE] Block {idx}: length={len(block)}")  # Log block index and length
            
            if len(block) < 80:  # Check if block is too short to be a valid suggestion
                logger.debug(f"[PARSE] Block {idx} too short ({len(block)} chars)")  # Log that block is being skipped
                continue  # Skip to next block
            
            # Check if block contains required fields
            has_file = 'FILE:' in block.upper()  # Check if block contains FILE: field (case-insensitive)
            has_change = 'CHANGE:' in block.upper()  # Check if block contains CHANGE: field
            has_code = 'CODE:' in block.upper()  # Check if block contains CODE: field
            has_why = 'WHY:' in block.upper()  # Check if block contains WHY: field
            
            logger.debug(f"[PARSE] Block {idx} fields: FILE={has_file}, CHANGE={has_change}, CODE={has_code}, WHY={has_why}")  # Log which fields were found
            
            # RELAXED: Accept suggestions with at least FILE, CODE, and one other field
            if has_file and has_code and (has_change or has_why):  # Check if block has minimum required fields (FILE, CODE, plus either CHANGE or WHY)
                suggestion = block.strip()  # Remove leading/trailing whitespace from block
                
                # Extract CODE field for deduplication
                code_match = re.search(r'CODE:\s*(.+?)(?:\n\s*(?:WHY:|CHANGE:|FILE:|$)|$)', suggestion, re.IGNORECASE | re.DOTALL)  # Regex to extract CODE field content (stops at next field or end)
                if code_match:  # If CODE field was successfully extracted
                    code_content = code_match.group(1).strip()  # Extract code content and remove whitespace
                    
                    logger.debug(f"[PARSE] Block {idx} CODE content: {code_content[:50]}...")  # Log first 50 chars of code for debugging
                    
                    # Create signature for deduplication
                    code_signature = code_content.lower().replace(' ', '').replace('"', '').replace("'", '')  # Normalize code by removing spaces and quotes (creates signature for comparison)
                    
                    # Check for duplicates
                    if code_signature in seen_codes:  # Check if we've already seen this code block
                        logger.debug(f"[PARSE] Block {idx} is duplicate")  # Log that block is a duplicate
                        continue  # Skip this duplicate suggestion
                    
                    # Quality check: ensure CODE has actual content
                    if len(code_content) > 10:  # Check if code content is substantial (> 10 chars, lowered threshold)
                        # Format nicely
                        suggestion = re.sub(r'FILE:\s*', '\n  FILE: ', suggestion, flags=re.IGNORECASE)  # Add newline and indent before FILE: field
                        suggestion = re.sub(r'CHANGE:\s*', '\n  CHANGE: ', suggestion, flags=re.IGNORECASE)  # Add newline and indent before CHANGE: field
                        suggestion = re.sub(r'CODE:\s*', '\n  CODE: ', suggestion, flags=re.IGNORECASE)  # Add newline and indent before CODE: field
                        suggestion = re.sub(r'WHY:\s*', '\n  WHY: ', suggestion, flags=re.IGNORECASE)  # Add newline and indent before WHY: field
                        
                        seen_codes.add(code_signature)  # Mark this code signature as seen (prevents duplicates)
                        suggestions.append(suggestion)  # Add formatted suggestion to list
                        logger.info(f"[PARSE] ✅ Accepted block {idx}")  # Log successful acceptance of suggestion
                    else:  # Code content is too short
                        logger.debug(f"[PARSE] Block {idx} CODE too short: {len(code_content)} chars")  # Log why block was rejected
                else:  # Failed to extract CODE field
                    logger.debug(f"[PARSE] Block {idx} failed to extract CODE field")  # Log parsing failure
            else:  # Block is missing required fields
                missing = []  # Initialize list to track which fields are missing
                if not has_file: missing.append("FILE")  # Add FILE to missing list if not found
                if not has_change: missing.append("CHANGE")  # Add CHANGE to missing list if not found
                if not has_code: missing.append("CODE")  # Add CODE to missing list if not found
                if not has_why: missing.append("WHY")  # Add WHY to missing list if not found
                logger.debug(f"[PARSE] Block {idx} missing fields: {', '.join(missing)}")  # Log which fields are missing
        
        if not suggestions:  # Check if no suggestions were successfully parsed
            logger.error("[PARSE] No valid suggestions found after parsing all blocks")  # Log parsing failure
        else:  # Successfully parsed at least one suggestion
            logger.info(f"[PARSE] Successfully parsed {len(suggestions)} suggestions")  # Log success with count
        
        return suggestions  # Return list of parsed and formatted suggestions
>>>>>>> Stashed changes
