"""Enhanced RAG analysis with workspace integration"""  # Module docstring describing this file combines vectorstore retrieval, workspace analysis, and LLM suggestions for comprehensive CI/CD predictions
import logging  # Standard Python logging library for tracking analysis pipeline
import re  # Regular expression library for pattern matching in rules and pipeline content
import asyncio  # Python's async library for concurrent operations (used for LLM calls)
from typing import Dict, List  # Type hints for function signatures: Dict for dictionaries, List for arrays
from app.services.workspace_analysis_service import WorkspaceAnalysisService  # Service that scans local Yocto workspace for configuration rules
from app.services.retrieval_service import RetrievalService  # Service that queries vectorstore for relevant documents
from app.services.vectorstore_service import VectorStoreService  # Service that manages FAISS vectorstore operations
from app.extractors.rule_extractor import RuleExtractor  # Extractor that pulls rules from text using linguistic/structural patterns
from app.core.config import get_settings  # Function to load application configuration settings



logger = logging.getLogger(__name__)  # Create logger instance for this module to output analysis pipeline steps



class EnhancedAnalysisService:  # Main analysis service that orchestrates RAG pipeline, rule extraction, evaluation, and prediction
    """Advanced RAG analysis with configurable prediction logic"""  # Docstring explaining this service provides comprehensive AI-driven build prediction
    
    def __init__(self):  # Constructor that initializes all required services and components
        try:  # Wrap initialization in try-except to catch setup errors
            logger.debug("[INIT] Initializing EnhancedAnalysisService...")  # Log start of initialization
            
            # Load settings for configurable thresholds
            self.settings = get_settings()  # Load application settings (thresholds, paths, etc.) from config
            logger.debug("[OK] Settings loaded")  # Log successful settings load
            
            self.rule_extractor = RuleExtractor()  # Initialize rule extractor for finding patterns in documents
            logger.debug("[OK] RuleExtractor initialized")  # Log successful rule extractor creation
            
            self.workspace_service = WorkspaceAnalysisService()  # Initialize workspace scanner for local Yocto project analysis
            logger.debug("[OK] WorkspaceAnalysisService initialized")  # Log successful workspace service creation
            
            self.vectorstore_service = VectorStoreService(vector_store_path="./vectorstore")  # Initialize vectorstore service with path to FAISS index files
            logger.debug("[OK] VectorStoreService initialized")  # Log successful vectorstore service creation
            logger.info(f"    Vectorstore loaded: {self.vectorstore_service.is_loaded()}")  # Log whether vectorstore index was successfully loaded
            
            self.retrieval_service = RetrievalService(self.vectorstore_service)  # Initialize retrieval service with vectorstore reference for querying
            logger.debug("[OK] RetrievalService initialized")  # Log successful retrieval service creation
            
            logger.info("[OK] EnhancedAnalysisService fully initialized")  # Log that all components are ready
        except Exception as e:  # Catch any initialization errors
            logger.exception(f"[ERROR] Error initializing EnhancedAnalysisService: {e}")  # Log full exception traceback for debugging
            raise  # Re-raise exception so caller knows initialization failed
    
    async def comprehensive_analyze(self, pipeline_content: str, user_id: str = None) -> Dict:  # Main analysis method that runs entire RAG pipeline asynchronously
        """Full analysis"""  # Docstring describing this method performs complete analysis from retrieval to prediction
        
        try:  # Wrap entire analysis in try-except to handle errors gracefully
            logger.info(f"[ANALYSIS] Starting comprehensive analysis for user: {user_id}")  # Log start of analysis with user identifier
            
            # Step 1: Get docs from vectorstore
            relevant_docs = []  # Initialize empty list to store retrieved documents
            try:  # Wrap retrieval in try-except to continue even if vectorstore fails
                logger.debug("[RETRIEVE] Retrieving documents from vectorstore...")  # Log start of document retrieval
                relevant_docs = self.retrieval_service.retrieve_relevant_documents(pipeline_content, k=10)  # Query vectorstore for 10 most relevant documents based on pipeline content similarity
                logger.info(f"[OK] Retrieved {len(relevant_docs)} documents")  # Log number of documents successfully retrieved
            except Exception as e:  # Catch retrieval errors
                logger.exception(f"[ERROR] Error retrieving documents: {e}")  # Log error with full traceback
            
            # Step 2: Extract rules from EACH document using extract_all_rules() 
            vectorstore_rules = []  # Initialize empty list to collect all rules from retrieved documents
            for doc in relevant_docs:  # Loop through each retrieved document
                try:  # Wrap extraction in try-except to skip problematic documents
                    rules = self.rule_extractor.extract_all_rules(doc.page_content)  # Extract all rule types (linguistic, structural, procedural, etc.) from document text
                    vectorstore_rules.extend(rules)  # Add extracted rules to master list
                    logger.debug(f"[EXTRACT] Extracted {len(rules)} rules from document")  # Log number of rules extracted from this document
                except Exception as e:  # Catch extraction errors
                    logger.debug(f"[WARN] Error extracting from doc: {e}")  # Log warning but continue processing other documents
            
            logger.info(f"[OK] Extracted {len(vectorstore_rules)} rules from vectorstore")  # Log total rules extracted from all vectorstore documents
            
            # Step 3: Extract workspace rules
            workspace_rules = []  # Initialize empty list for rules extracted from local workspace
            workspaces = []  # Initialize empty list to store discovered workspace paths
            try:  # Wrap workspace analysis in try-except since workspace may not exist
                logger.debug("[SCAN] Discovering Yocto workspace...")  # Log start of workspace discovery
                workspaces = self.workspace_service.discover_yocto_workspace()  # Scan filesystem for Yocto project directories (looks for conf/, layers/, etc.)
                logger.info(f"[OK] Found {len(workspaces)} workspace(s)")  # Log number of Yocto workspaces discovered
                
                if workspaces:  # If at least one workspace was found
                    for ws in workspaces[:1]:  # Process only the first workspace ([:1] limits to first element)
                        rules = self.workspace_service.extract_rules_from_workspace(ws)  # Extract Yocto-specific rules from local.conf, bblayers.conf, recipes, etc.
                        workspace_rules.extend(rules)  # Add workspace rules to list
                        logger.info(f"[OK] Extracted {len(rules)} workspace rules")  # Log number of rules extracted from this workspace
            except Exception as e:  # Catch workspace errors (e.g., no workspace found, permission issues)
                logger.warning(f"[WARN] Workspace error: {e}")  # Log warning but continue with analysis using only vectorstore rules
            
            # Step 4: Combine all rules
            all_rules = vectorstore_rules + workspace_rules  # Combine rules from vectorstore documents and local workspace into single list
            try:  # Wrap deduplication in try-except
                all_rules = self.rule_extractor._deduplicate(all_rules)  # Remove duplicate rules by comparing rule text (avoids redundant evaluation)
                logger.info(f"[OK] Combined {len(all_rules)} unique rules")  # Log total unique rules after deduplication
            except Exception as e:  # Catch deduplication errors
                logger.warning(f"[WARN] Dedup error: {e}")  # Log warning but continue with possibly duplicated rules
            
            if len(all_rules) == 0:  # Check if no rules were extracted at all
                logger.warning("[WARN] NO RULES EXTRACTED")  # Log warning that analysis will be limited without rules
            
            # Step 5: Evaluate rules
            result = {'satisfied': 0, 'violated': 0, 'violations': []}  # Initialize result dictionary with default values
            try:  # Wrap evaluation in try-except
                result = self._evaluate_rules(all_rules, pipeline_content)  # Check each rule against pipeline content to determine satisfaction status
                logger.info(f"[EVAL] {result.get('satisfied')}/{len(all_rules)} satisfied")  # Log how many rules were satisfied
                logger.info(f"[VIOLATIONS] {result.get('violated')} violated")  # Log how many rules were violated
            except Exception as e:  # Catch evaluation errors
                logger.exception(f"[ERROR] Evaluation error: {e}")  # Log error with full traceback
            
            # Step 6: Make prediction using configurable thresholds
            prediction = {'outcome': 'UNKNOWN', 'confidence': 50}  # Initialize prediction with default unknown state
            try:  # Wrap prediction in try-except
                prediction = self._make_prediction(result, len(all_rules))  # Calculate prediction (PASS/HIGH_RISK/FAIL) based on rule satisfaction ratios and configured thresholds
                logger.info(f"[OK] Prediction: {prediction['outcome']} ({prediction['confidence']}%)")  # Log final prediction outcome and confidence score
            except Exception as e:  # Catch prediction errors
                logger.exception(f"[ERROR] Prediction error: {e}")  # Log error with full traceback
            
            # Step 7: Generate AI suggestions with WORKSPACE PATH
            ai_suggestions = []  # Initialize empty list for LLM-generated suggestions
            try:  # Wrap LLM interaction in try-except since LLM may not be available
                from app.services.llm_service import LLMService  # Import LLM service (lazy import to avoid circular dependencies)
                logger.debug("[LLM] Initializing LLM service...")  # Log start of LLM initialization
                llm_service = LLMService()  # Create LLM service instance (connects to Ollama or other LLM backend)
                
                if llm_service.llm is not None and result.get('violated') > 0:  # Only generate suggestions if LLM is available and there are violations to address
                    # Get violated rules
                    violated_rule_details = [  # Build list of rules that were violated (failed satisfaction check)
                        r for r in all_rules 
                        if not self._rule_satisfied(  # Filter for rules where satisfaction check returns False
                            r.get('rule_text', '') if isinstance(r, dict) else str(r),  # Extract rule text (handle both dict and string formats)
                            pipeline_content  # Check against pipeline content
                        )
                    ]
                    
                    logger.info(f"[VIOLATIONS] Found {len(violated_rule_details)} violations")  # Log total number of violated rules
                    
                    if violated_rule_details:  # If there are violated rules to process
                        for i, rule in enumerate(violated_rule_details[:5]):  # Loop through first 5 violations for logging ([:5] limits output)
                            rule_text = rule.get('rule_text', str(rule))[:80]  # Extract rule text and truncate to 80 characters for readable logging
                            logger.info(f"    {i+1}. {rule_text}")  # Log each violation with number
                        
                        logger.info(f"[LLM] Generating suggestions for {len(violated_rule_details)} violations...")  # Log start of LLM suggestion generation
                        
                        # Get workspace path for file scanning
                        workspace_path = workspaces[0] if workspaces else None  # Use first workspace path if available, otherwise None (LLM can scan actual files for context)
                        logger.info(f"[LLM] Using workspace path: {workspace_path}")  # Log workspace path being passed to LLM
                        
                        # Call LLM with workspace path for ultra-specific suggestions
                        ai_suggestions = await llm_service.generate_suggestions(  # Call async LLM service to generate actionable suggestions (await since it's async)
                            violated_rule_details[:7],  # Pass first 7 violated rules (limit to avoid overwhelming LLM context)
                            pipeline_content,  # Pass pipeline content for context
                            workspace_path  # Pass workspace path so LLM can reference actual config files
                        )
                        
                        if ai_suggestions:  # If LLM returned suggestions
                            logger.info(f"[OK] Got {len(ai_suggestions)} AI suggestions")  # Log number of suggestions received
                        else:  # If LLM returned empty list
                            logger.warning("[WARN] LLM returned no suggestions")  # Log warning that LLM didn't generate any suggestions
                    else:  # If violated_rule_details is empty (shouldn't happen if violated > 0, but defensive check)
                        logger.info("[OK] No violations detected")  # Log that no violations were found
                else:  # If LLM is not available or no violations exist
                    logger.info("[OK] No violations or LLM not initialized")  # Log why suggestions weren't generated
            except Exception as e:  # Catch any LLM-related errors
                logger.warning(f"[WARN] AI error: {e}")  # Log warning with error message
                import traceback  # Import traceback module for detailed error logging
                traceback.print_exc()  # Print full exception traceback to console for debugging
            
            # Step 8: Build response
            analysis_text = ""  # Initialize empty string for final analysis response text
            try:  # Wrap response building in try-except
                logger.debug("[BUILD] Building response...")  # Log start of response building
                analysis_text = self._build_comprehensive_response(  # Build formatted text response with all analysis results
                    pipeline_content,  # Original pipeline content
                    all_rules,  # Combined list of all extracted rules
                    result,  # Evaluation results (satisfied/violated counts)
                    relevant_docs,  # Retrieved vectorstore documents for sources section
                    workspace_rules,  # Rules from local workspace
                    prediction,  # Prediction outcome and confidence
                    ai_suggestions  # LLM-generated suggestions
                )
                logger.info(f"[OK] Response built ({len(analysis_text)} chars)")  # Log successful response building with character count
                
                # Add pattern alerts from learned feedback
                analysis_text = self.add_pattern_alerts(pipeline_content, analysis_text)  # Append learned pattern warnings from historical feedback data
                logger.info("[OK] Pattern alerts added to analysis")  # Log that pattern alerts were successfully added
                
            except Exception as e:  # Catch response building errors
                logger.exception(f"[ERROR] Response error: {e}")  # Log error with full traceback
                analysis_text = f"Error: {str(e)}"  # Set analysis text to error message as fallback
            
            return {  # Return comprehensive analysis result dictionary
                'prediction': prediction['outcome'],  # Predicted outcome (PASS/HIGH_RISK/FAIL)
                'confidence': prediction['confidence'],  # Confidence score (0-100)
                'analysis': analysis_text,  # Full formatted analysis text with all findings
                'risk_factors': result.get('violations', []),  # List of violated rules (risk factors)
                'satisfied_rules': result.get('satisfied', 0),  # Count of satisfied rules
                'violated_rules': result.get('violated', 0),  # Count of violated rules
                'workspace_found': len(workspace_rules) > 0,  # Boolean indicating if workspace was found and analyzed
                'total_rules': len(all_rules),  # Total number of unique rules evaluated
                'vectorstore_docs_used': len(relevant_docs),  # Number of documents retrieved from vectorstore
                'all_rules': all_rules,  # Complete list of all rules (for potential frontend display)
                'stack': ['BitBake', 'Yocto', 'Jenkins', 'Bash']  # Detected technology stack (hardcoded for Yocto context)
            }
        
        except Exception as e:  # Catch any unexpected errors in the entire analysis pipeline
            logger.exception(f"[CRITICAL ERROR] {e}")  # Log critical error with full traceback
            import traceback  # Import traceback for detailed error logging
            traceback.print_exc()  # Print full exception traceback to console
            return {  # Return error result dictionary with safe default values
                'prediction': 'ERROR',  # Set prediction to ERROR state
                'confidence': 0,  # Zero confidence due to error
                'analysis': f'Error: {str(e)}',  # Include error message in analysis
                'risk_factors': [],  # Empty risk factors list
                'satisfied_rules': 0,  # Zero satisfied rules
                'violated_rules': 0,  # Zero violated rules
                'workspace_found': False,  # No workspace found
                'total_rules': 0,  # Zero total rules
                'vectorstore_docs_used': 0,  # No documents used
                'all_rules': [],  # Empty rules list
                'stack': []  # Empty stack list
            }
    
    def add_pattern_alerts(self, pipeline_content: str, analysis: str) -> str:  # Method to append learned pattern warnings based on historical feedback data
        """Smart pattern matching with keyword extraction from learned feedback"""  # Docstring explaining this uses fuzzy matching on historical patterns
        alerts = []  # Initialize empty list to collect pattern alerts
        
        try:  # Wrap pattern matching in try-except to handle database errors
            from app.core.database import get_db_connection, get_db_cursor  # Import database utilities for querying learned patterns
            import re  # Import regex for keyword extraction
            
            # Common words to ignore in keyword matching
            STOP_WORDS = {  # Set of common English words to exclude from keyword matching (improves fuzzy matching quality)
                'the', 'and', 'for', 'with', 'this', 'that', 'from', 'has', 
                'are', 'was', 'been', 'have', 'had', 'will', 'can', 'should'
            }
            
            with get_db_connection() as conn:  # Open database connection using context manager (auto-cleanup)
                with get_db_cursor(conn) as cur:  # Open database cursor using context manager
                    cur.execute("""
                        SELECT pattern_text, pattern_type, occurrences
                        FROM learned_patterns
                        WHERE occurrences >= 1
                        ORDER BY occurrences DESC
                    """)  # Query learned_patterns table for patterns that have occurred at least once (ordered by most common first)
                    
                    for row in cur.fetchall():  # Loop through each learned pattern from database
                        pattern_text = row['pattern_text']  # Extract pattern text from database row
                        pipeline_lower = pipeline_content.lower()  # Convert pipeline content to lowercase for case-insensitive matching
                        
                        # Method 1: Exact match (fastest)
                        if pattern_text.lower() in pipeline_lower:  # Check if pattern appears exactly in pipeline content (substring match)
                            alerts.append({  # Add exact match alert with 100% confidence
                                'text': pattern_text,  # The pattern text that matched
                                'count': row['occurrences'],  # How many times this pattern has been reported in feedback
                                'confidence': 100  # 100% confidence for exact match
                            })
                            continue  # Skip fuzzy matching since we have exact match
                        
                        # Method 2: Keyword matching (for imperfect user input)
                        keywords = [  # Extract significant keywords from pattern text (3+ character words, excluding stop words)
                            w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', pattern_text)  # Find words with 3+ alphanumeric/underscore/hyphen characters
                            if w.lower() not in STOP_WORDS  # Exclude common stop words
                        ]
                        
                        if keywords:  # If we extracted any meaningful keywords
                            matches = sum(1 for kw in keywords if kw in pipeline_lower)  # Count how many keywords appear in pipeline content
                            match_ratio = matches / len(keywords)  # Calculate percentage of keywords that matched
                            
                            # 50%+ keyword match = show alert
                            if match_ratio >= 0.5:  # If at least 50% of keywords matched (fuzzy match threshold)
                                alerts.append({  # Add fuzzy match alert
                                    'text': pattern_text,  # The pattern text that partially matched
                                    'count': row['occurrences'],  # Occurrence count from database
                                    'confidence': int(match_ratio * 100)  # Confidence based on keyword match ratio (50-99%)
                                })
                                logger.info(  # Log fuzzy match details for debugging
                                    f"[PATTERN] Fuzzy match: {pattern_text[:40]} "  # Show first 40 chars of pattern
                                    f"({matches}/{len(keywords)} keywords)"  # Show match ratio in readable format
                                )
        
        except Exception as e:  # Catch any errors during pattern matching
            logger.error(f"Pattern matching error: {e}")  # Log error (but don't fail entire analysis)
        
        if alerts:  # If we found any pattern alerts
            # Sort by confidence and occurrence count
            alerts.sort(key=lambda x: (x['count'], x['confidence']), reverse=True)  # Sort alerts by occurrence count first, then confidence (descending order)
            
            alert_lines = [  # Format alerts as warning messages
                f"  WARNING: {alert['text']} (reported {alert['count']}x)"  # Show pattern text and how many times it's been reported
                for alert in alerts[:5]  # Take only top 5 matches ([:5] limits output)
            ]
            
            alert_section = "\n\nLEARNED PATTERN ALERTS:\n" + "\n".join(alert_lines)  # Build formatted alert section with header
            return analysis + alert_section  # Append alert section to existing analysis text
        
        return analysis  # Return analysis unchanged if no alerts found
    
    def _evaluate_rules(self, rules: List[Dict], pipeline: str) -> Dict:  # Method to check each rule against pipeline content and count satisfied/violated
        """Evaluate rules against pipeline content"""  # Docstring explaining this performs rule-by-rule evaluation
        satisfied = 0  # Counter for rules that passed satisfaction check
        violated = 0  # Counter for rules that failed satisfaction check
        violations = []  # List to store violated rule texts for display
        
        for rule in rules:  # Loop through each rule in the list
            rule_text = rule.get('rule_text', '') if isinstance(rule, dict) else str(rule)  # Extract rule text (handle both dict and string formats)
            if self._rule_satisfied(rule_text, pipeline):  # Check if rule is satisfied using keyword matching
                satisfied += 1  # Increment satisfied counter
            else:  # Rule is not satisfied
                violated += 1  # Increment violated counter
                violations.append(rule_text[:100])  # Add truncated rule text to violations list ([:100] limits to 100 chars)
        
        return {  # Return evaluation results dictionary
            'satisfied': satisfied,  # Total number of satisfied rules
            'violated': violated,  # Total number of violated rules
            'violations': violations[:10]  # First 10 violation texts for display ([:10] limits output)
        }
    
    def _rule_satisfied(self, rule: str, pipeline: str) -> bool:  # Method to check if a single rule is satisfied based on keyword matching
        """
        Check if rule is satisfied using configurable keyword threshold
        
        Args:
            rule: Rule text to check
            pipeline: Pipeline content to check against
        
        Returns:
            bool: True if rule is satisfied based on keyword matching
        """  # Detailed docstring explaining satisfaction logic
        if not rule or not pipeline:  # Check if either input is empty/None
            return False  # Return False for invalid inputs (can't evaluate)
        
        rule_lower = rule.lower()  # Convert rule to lowercase for case-insensitive matching
        pipeline_lower = pipeline.lower()  # Convert pipeline to lowercase for case-insensitive matching
        
        # Extract key terms (3+ character words)
        key_terms = re.findall(r'\b[a-z]{3,}\b', rule_lower)  # Find all words with 3+ lowercase letters (significant keywords)
        
        if not key_terms:  # If no key terms extracted (rule is too short or has no meaningful words)
            return False  # Return False since we can't evaluate satisfaction
        
        # Count how many key terms appear in pipeline
        matches = sum(1 for term in key_terms if term in pipeline_lower)  # Count how many key terms from rule appear in pipeline content
        match_ratio = matches / len(key_terms)  # Calculate percentage of key terms that matched
        
        # Use configurable threshold from settings
        threshold = self.settings.rule_satisfaction_keyword_threshold  # Load satisfaction threshold from config (e.g., 0.4 = 40% of keywords must match)
        is_satisfied = match_ratio >= threshold  # Rule is satisfied if match ratio meets or exceeds threshold
        
        logger.debug(  # Log satisfaction check details for debugging
            f"[RULE] '{rule[:50]}...' match ratio: {match_ratio:.2%} "  # Show first 50 chars of rule and match ratio as percentage
            f"(threshold: {threshold:.0%}) = {'SATISFIED' if is_satisfied else 'VIOLATED'}"  # Show threshold and final result
        )
        
        return is_satisfied  # Return True if satisfied, False if violated
    
    def _make_prediction(self, result: Dict, total_rules: int) -> Dict:  # Method to calculate final prediction based on rule evaluation results
        """
        Make prediction based on RAW confidence 
        Outcome determined by confidence value
        
        Args:
            result: Evaluation results with satisfied/violated counts
            total_rules: Total number of rules evaluated
        
        Returns:
            Dict with 'outcome' and 'confidence' (0-100)
        """  # Detailed docstring explaining prediction logic
        
        # Handle no rules case
        if total_rules == 0:  # Check if no rules were extracted/evaluated
            if self.settings.prediction_unknown_on_no_rules:  # Check config flag for how to handle no-rules case
                logger.warning("[PREDICTION] No rules found - returning UNKNOWN")  # Log that we're returning unknown state
                return {'outcome': 'UNKNOWN', 'confidence': 50}  # Return unknown with neutral 50% confidence
            else:  # Config says default to pass when no rules exist
                logger.warning("[PREDICTION] No rules found - defaulting to PASS")  # Log that we're defaulting to pass
                return {'outcome': 'PASS', 'confidence': 70}  # Return pass with moderate 70% confidence
        
        satisfied = result.get('satisfied', 0)  # Extract satisfied count from evaluation results (default 0 if missing)
        violated = result.get('violated', 0)  # Extract violated count from evaluation results (default 0 if missing)
        
        # Calculate satisfaction ratio (internal only)
        satisfaction_ratio = satisfied / total_rules  # Calculate percentage of rules that were satisfied (0.0 to 1.0)
        violation_ratio = violated / total_rules  # Calculate percentage of rules that were violated (0.0 to 1.0)
        
        # Calculate confidence from satisfaction
        # More satisfied rules = higher confidence
        if violation_ratio < self.settings.prediction_high_risk_violation_threshold:  # Check if violation rate is below high-risk threshold (e.g., < 0.3 = less than 30% violations)
            # Low violations: boost confidence
            confidence = int(satisfaction_ratio * 100) + self.settings.prediction_pass_confidence_boost  # Convert satisfaction ratio to percentage and add confidence boost (e.g., +10)
            confidence = min(100, confidence)  # Cap confidence at 100 (can't exceed 100%)
        else:  # Violation rate is moderate to high
            # Moderate to high violations: raw satisfaction
            confidence = int(satisfaction_ratio * 100)  # Use raw satisfaction percentage without boost
        
        # Determine outcome based on confidence ranges
        if confidence <= self.settings.prediction_fail_max:  # Check if confidence is in fail range (e.g., <= 40)
            outcome = 'FAIL'  # Predict build will fail
        elif confidence <= self.settings.prediction_high_risk_max:  # Check if confidence is in high-risk range (e.g., <= 70)
            outcome = 'HIGH_RISK'  # Predict build is high risk (may fail)
        else:  # Confidence is above high-risk threshold
            outcome = 'PASS'  # Predict build will pass
        
        logger.info(f"[PREDICTION] {outcome} @ {confidence}%")  # Log final prediction with confidence score
        
        return {  # Return prediction dictionary
            'outcome': outcome,  # Predicted outcome (PASS/HIGH_RISK/FAIL)
            'confidence': confidence  # Confidence score (0-100)
        }


    
    def _build_comprehensive_response(self, pipeline, rules, result, docs, workspace_rules, prediction, ai_suggestions=None) -> str:  # Method to build formatted analysis response text with all findings
        """Build comprehensive analysis response with all findings"""  # Docstring explaining this creates human-readable analysis report
        
        response = f"""


ESTABLISHED RULES ANALYSIS:
{'='*70}
"""  # Initialize response with header section (70 equals signs for visual separation)
        
        # Show first 50 rules with their status
        for i, rule in enumerate(rules[:50], 1):  # Loop through first 50 rules (enumerate starts at 1 for human-readable numbering)
            rule_text = rule.get('rule_text', '')[:100] if isinstance(rule, dict) else str(rule)[:100]  # Extract rule text and truncate to 100 chars (handle dict/string formats)
            status = "PASS" if self._rule_satisfied(rule_text, pipeline) else "FAIL"  # Check if rule is satisfied and set status string
            response += f"{i}. {rule_text}... - {status}\n"  # Add numbered rule with status to response
        
        # Show workspace-specific findings
        if workspace_rules:  # If workspace rules were extracted
            response += f"\nREAL YOCTO WORKSPACE ANALYSIS:\n"  # Add workspace section header
            response += f"Found {len(workspace_rules)} specific rules from actual Yocto files in your workspace\n"  # Show count of workspace rules
            for r in workspace_rules[:10]:  # Loop through first 10 workspace rules ([:10] limits output)
                rule_text = r.get('rule_text', '') if isinstance(r, dict) else str(r)  # Extract rule text (handle dict/string formats)
                response += f"  * {rule_text}\n"  # Add bullet point with rule text (indented for visual hierarchy)
        
        response += f"""
APPLICABLE_RULES: {len(rules)}
SATISFIED_RULES: {result.get('satisfied', 0)}
VIOLATED_RULES: {result.get('violated', 0)}


{'='*70}
BUILD PREDICTION: {prediction['outcome']}
CONFIDENCE: {prediction['confidence']}%


AI-GENERATED SUGGESTIONS TO IMPROVE CONFIDENCE:
"""  # Add summary section with counts, prediction, and suggestions header
        
        # Add AI suggestions
        if ai_suggestions and len(ai_suggestions) > 0:  # Check if LLM generated any suggestions
            logger.info(f"[BUILD] Adding {len(ai_suggestions)} AI suggestions")  # Log that we're adding suggestions
            for i, suggestion in enumerate(ai_suggestions, 1):  # Loop through suggestions with numbering (starts at 1)
                response += f"{i}. {suggestion}\n"  # Add numbered suggestion to response
        else:  # No suggestions were generated
            logger.warning("[WARN] No AI suggestions generated")  # Log warning that suggestions are missing
            response += "No specific suggestions generated for this analysis.\n"  # Add fallback message
        
        # Add sources section
        response += "\n" + "="*70 + "\nSOURCES & REFERENCES:\n" + "="*70 + "\n"  # Add sources section header with separator lines
        
        seen_sources = set()  # Initialize set to track sources we've already added (prevents duplicates)
        sources_list = []  # Initialize list to store formatted source lines
        source_count = 1  # Counter for source numbering (starts at 1)
        
        if docs and len(docs) > 0:  # Check if we have retrieved documents to cite
            logger.info(f"[BUILD] Processing {len(docs)} documents...")  # Log that we're processing documents for sources
            for doc in docs[:20]:  # Loop through first 20 documents ([:20] limits source count)
                try:  # Wrap source processing in try-except to skip problematic documents
                    if hasattr(doc, 'metadata') and doc.metadata:  # Check if document has metadata attribute
                        # Confluence sources
                        confluence_url = doc.metadata.get('confluence_url', '')  # Extract Confluence URL from metadata (empty string if missing)
                        if confluence_url:  # If document came from Confluence
                            source = doc.metadata.get('source', 'Confluence')  # Extract source field (default to 'Confluence')
                            unique_key = f"confluence:{confluence_url}"  # Create unique key for deduplication
                            
                            if unique_key not in seen_sources:  # Check if we haven't already added this source
                                seen_sources.add(unique_key)  # Mark this source as seen
                                page_name = source.split('/')[-1].replace('.md', '').replace('_', ' ') if '/' in source else 'Confluence'  # Extract page name from path (clean up .md and underscores)
                                sources_list.append(f"{source_count}. Confluence: {page_name} - {confluence_url}")  # Add formatted Confluence source with number
                                source_count += 1  # Increment source counter
                        
                        # Jenkins sources
                        source = doc.metadata.get('source', '')  # Extract source field from metadata
                        if 'jenkins' in str(source).lower():  # Check if source contains 'jenkins' (case-insensitive)
                            match = re.search(r'([a-zA-Z0-9_-]+)-(\d+)', str(source))  # Regex to extract job name and build number (e.g., "my-job-123")
                            if match:  # If regex found job and build number
                                job, build = match.groups()  # Extract job name and build number from regex groups
                                unique_key = f"jenkins:{job}:{build}"  # Create unique key for deduplication
                                
                                if unique_key not in seen_sources:  # Check if we haven't already added this source
                                    seen_sources.add(unique_key)  # Mark this source as seen
                                    jenkins_url = f"{job}/{build}"  # Format Jenkins URL path
                                    sources_list.append(f"{source_count}. Jenkins: {job} #{build} - {jenkins_url}")  # Add formatted Jenkins source with number
                                    source_count += 1  # Increment source counter
                except Exception as e:  # Catch any errors processing individual sources
                    logger.warning(f"[WARN] Error processing source: {e}")  # Log warning but continue processing other sources
        
        if sources_list:  # If we collected any sources
            logger.info(f"[OK] Showing {len(sources_list)} UNIQUE sources")  # Log number of unique sources being displayed
            for source in sources_list:  # Loop through formatted source lines
                response += f"{source}\n"  # Add source line to response
        else:  # No sources were found
            logger.warning("[WARN] No sources found")  # Log warning that no sources are available
            response += "No specific sources retrieved for this analysis.\n"  # Add fallback message
        
        response += "="*70 + "\n"  # Add closing separator line
        
        return response  # Return complete formatted response text
