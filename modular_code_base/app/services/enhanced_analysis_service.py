"""Enhanced RAG analysis with workspace integration"""
import logging
import re
import asyncio
from typing import Dict, List
from app.services.workspace_analysis_service import WorkspaceAnalysisService
from app.services.retrieval_service import RetrievalService
from app.services.vectorstore_service import VectorStoreService
from app.extractors.rule_extractor import RuleExtractor
from app.core.config import get_settings


logger = logging.getLogger(__name__)


class EnhancedAnalysisService:
    """Advanced RAG analysis with configurable prediction logic"""
    
    def __init__(self):
        try:
            logger.debug("[INIT] Initializing EnhancedAnalysisService...")
            
            # Load settings for configurable thresholds
            self.settings = get_settings()
            logger.debug("[OK] Settings loaded")
            
            self.rule_extractor = RuleExtractor()
            logger.debug("[OK] RuleExtractor initialized")
            
            self.workspace_service = WorkspaceAnalysisService()
            logger.debug("[OK] WorkspaceAnalysisService initialized")
            
            self.vectorstore_service = VectorStoreService(vector_store_path="./vectorstore")
            logger.debug("[OK] VectorStoreService initialized")
            logger.info(f"    Vectorstore loaded: {self.vectorstore_service.is_loaded()}")
            
            self.retrieval_service = RetrievalService(self.vectorstore_service)
            logger.debug("[OK] RetrievalService initialized")
            
            logger.info("[OK] EnhancedAnalysisService fully initialized")
        except Exception as e:
            logger.exception(f"[ERROR] Error initializing EnhancedAnalysisService: {e}")
            raise
    
    async def comprehensive_analyze(self, pipeline_content: str, user_id: str = None) -> Dict:
        """Full analysis"""
        
        try:
            logger.info(f"[ANALYSIS] Starting comprehensive analysis for user: {user_id}")
            
            # Step 1: Get docs from vectorstore
            relevant_docs = []
            try:
                logger.debug("[RETRIEVE] Retrieving documents from vectorstore...")
                relevant_docs = self.retrieval_service.retrieve_relevant_documents(pipeline_content, k=10)
                logger.info(f"[OK] Retrieved {len(relevant_docs)} documents")
            except Exception as e:
                logger.exception(f"[ERROR] Error retrieving documents: {e}")
            
            # Step 2: Extract rules from EACH document using extract_all_rules() 
            vectorstore_rules = []
            for doc in relevant_docs:
                try:
                    rules = self.rule_extractor.extract_all_rules(doc.page_content)
                    vectorstore_rules.extend(rules)
                    logger.debug(f"[EXTRACT] Extracted {len(rules)} rules from document")
                except Exception as e:
                    logger.debug(f"[WARN] Error extracting from doc: {e}")
            
            logger.info(f"[OK] Extracted {len(vectorstore_rules)} rules from vectorstore")
            
            # Step 3: Extract workspace rules
            workspace_rules = []
            workspaces = []
            try:
                logger.debug("[SCAN] Discovering Yocto workspace...")
                workspaces = self.workspace_service.discover_yocto_workspace()
                logger.info(f"[OK] Found {len(workspaces)} workspace(s)")
                
                if workspaces:
                    for ws in workspaces[:1]:
                        rules = self.workspace_service.extract_rules_from_workspace(ws)
                        workspace_rules.extend(rules)
                        logger.info(f"[OK] Extracted {len(rules)} workspace rules")
            except Exception as e:
                logger.warning(f"[WARN] Workspace error: {e}")
            
            # Step 4: Combine all rules
            all_rules = vectorstore_rules + workspace_rules
            try:
                all_rules = self.rule_extractor._deduplicate(all_rules)
                logger.info(f"[OK] Combined {len(all_rules)} unique rules")
            except Exception as e:
                logger.warning(f"[WARN] Dedup error: {e}")
            
            if len(all_rules) == 0:
                logger.warning("[WARN] NO RULES EXTRACTED")
            
            # Step 5: Evaluate rules
            result = {'satisfied': 0, 'violated': 0, 'violations': []}
            try:
                result = self._evaluate_rules(all_rules, pipeline_content)
                logger.info(f"[EVAL] {result.get('satisfied')}/{len(all_rules)} satisfied")
                logger.info(f"[VIOLATIONS] {result.get('violated')} violated")
            except Exception as e:
                logger.exception(f"[ERROR] Evaluation error: {e}")
            
            # Step 6: Make prediction using configurable thresholds
            prediction = {'outcome': 'UNKNOWN', 'confidence': 50}
            try:
                prediction = self._make_prediction(result, len(all_rules))
                logger.info(f"[OK] Prediction: {prediction['outcome']} ({prediction['confidence']}%)")
            except Exception as e:
                logger.exception(f"[ERROR] Prediction error: {e}")
            
            # Step 7: Generate AI suggestions with WORKSPACE PATH
            ai_suggestions = []
            try:
                from app.services.llm_service import LLMService
                logger.debug("[LLM] Initializing LLM service...")
                llm_service = LLMService()
                
                if llm_service.llm is not None and result.get('violated') > 0:
                    # Get violated rules
                    violated_rule_details = [
                        r for r in all_rules 
                        if not self._rule_satisfied(
                            r.get('rule_text', '') if isinstance(r, dict) else str(r), 
                            pipeline_content
                        )
                    ]
                    
                    logger.info(f"[VIOLATIONS] Found {len(violated_rule_details)} violations")
                    
                    if violated_rule_details:
                        for i, rule in enumerate(violated_rule_details[:5]):
                            rule_text = rule.get('rule_text', str(rule))[:80]
                            logger.info(f"    {i+1}. {rule_text}")
                        
                        logger.info(f"[LLM] Generating suggestions for {len(violated_rule_details)} violations...")
                        
                        # Get workspace path for file scanning
                        workspace_path = workspaces[0] if workspaces else None
                        logger.info(f"[LLM] Using workspace path: {workspace_path}")
                        
                        # Call LLM with workspace path for ultra-specific suggestions
                        ai_suggestions = await llm_service.generate_suggestions(
                            violated_rule_details[:7],
                            pipeline_content,
                            workspace_path
                        )
                        
                        if ai_suggestions:
                            logger.info(f"[OK] Got {len(ai_suggestions)} AI suggestions")
                        else:
                            logger.warning("[WARN] LLM returned no suggestions")
                    else:
                        logger.info("[OK] No violations detected")
                else:
                    logger.info("[OK] No violations or LLM not initialized")
            except Exception as e:
                logger.warning(f"[WARN] AI error: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 8: Build response
            analysis_text = ""
            try:
                logger.debug("[BUILD] Building response...")
                analysis_text = self._build_comprehensive_response(
                    pipeline_content,
                    all_rules,
                    result,
                    relevant_docs,
                    workspace_rules,
                    prediction,
                    ai_suggestions
                )
                logger.info(f"[OK] Response built ({len(analysis_text)} chars)")
                
                # Add pattern alerts from learned feedback
                analysis_text = self.add_pattern_alerts(pipeline_content, analysis_text)
                logger.info("[OK] Pattern alerts added to analysis")
                
            except Exception as e:
                logger.exception(f"[ERROR] Response error: {e}")
                analysis_text = f"Error: {str(e)}"
            
            return {
                'prediction': prediction['outcome'],
                'confidence': prediction['confidence'],
                'analysis': analysis_text,
                'risk_factors': result.get('violations', []),
                'satisfied_rules': result.get('satisfied', 0),
                'violated_rules': result.get('violated', 0),
                'workspace_found': len(workspace_rules) > 0,
                'total_rules': len(all_rules),
                'vectorstore_docs_used': len(relevant_docs),
                'all_rules': all_rules,
                'stack': ['BitBake', 'Yocto', 'Jenkins', 'Bash']
            }
        
        except Exception as e:
            logger.exception(f"[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            return {
                'prediction': 'ERROR',
                'confidence': 0,
                'analysis': f'Error: {str(e)}',
                'risk_factors': [],
                'satisfied_rules': 0,
                'violated_rules': 0,
                'workspace_found': False,
                'total_rules': 0,
                'vectorstore_docs_used': 0,
                'all_rules': [],
                'stack': []
            }
    
    def add_pattern_alerts(self, pipeline_content: str, analysis: str) -> str:
        """Smart pattern matching with keyword extraction from learned feedback"""
        alerts = []
        
        try:
            from app.core.database import get_db_connection, get_db_cursor
            import re
            
            # Common words to ignore in keyword matching
            STOP_WORDS = {
                'the', 'and', 'for', 'with', 'this', 'that', 'from', 'has', 
                'are', 'was', 'been', 'have', 'had', 'will', 'can', 'should'
            }
            
            with get_db_connection() as conn:
                with get_db_cursor(conn) as cur:
                    cur.execute("""
                        SELECT pattern_text, pattern_type, occurrences
                        FROM learned_patterns
                        WHERE occurrences >= 1
                        ORDER BY occurrences DESC
                    """)
                    
                    for row in cur.fetchall():
                        pattern_text = row['pattern_text']
                        pipeline_lower = pipeline_content.lower()
                        
                        # Method 1: Exact match (fastest)
                        if pattern_text.lower() in pipeline_lower:
                            alerts.append({
                                'text': pattern_text,
                                'count': row['occurrences'],
                                'confidence': 100
                            })
                            continue
                        
                        # Method 2: Keyword matching (for imperfect user input)
                        keywords = [
                            w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', pattern_text)
                            if w.lower() not in STOP_WORDS
                        ]
                        
                        if keywords:
                            matches = sum(1 for kw in keywords if kw in pipeline_lower)
                            match_ratio = matches / len(keywords)
                            
                            # 50%+ keyword match = show alert
                            if match_ratio >= 0.5:
                                alerts.append({
                                    'text': pattern_text,
                                    'count': row['occurrences'],
                                    'confidence': int(match_ratio * 100)
                                })
                                logger.info(
                                    f"[PATTERN] Fuzzy match: {pattern_text[:40]} "
                                    f"({matches}/{len(keywords)} keywords)"
                                )
        
        except Exception as e:
            logger.error(f"Pattern matching error: {e}")
        
        if alerts:
            # Sort by confidence and occurrence count
            alerts.sort(key=lambda x: (x['count'], x['confidence']), reverse=True)
            
            alert_lines = [
                f"  WARNING: {alert['text']} (reported {alert['count']}x)"
                for alert in alerts[:5]  # Top 5 matches
            ]
            
            alert_section = "\n\nLEARNED PATTERN ALERTS:\n" + "\n".join(alert_lines)
            return analysis + alert_section
        
        return analysis
    
    def _evaluate_rules(self, rules: List[Dict], pipeline: str) -> Dict:
        """Evaluate rules against pipeline content"""
        satisfied = 0
        violated = 0
        violations = []
        
        for rule in rules:
            rule_text = rule.get('rule_text', '') if isinstance(rule, dict) else str(rule)
            if self._rule_satisfied(rule_text, pipeline):
                satisfied += 1
            else:
                violated += 1
                violations.append(rule_text[:100])
        
        return {
            'satisfied': satisfied,
            'violated': violated,
            'violations': violations[:10]
        }
    
    def _rule_satisfied(self, rule: str, pipeline: str) -> bool:
        """
        Check if rule is satisfied using configurable keyword threshold
        
        Args:
            rule: Rule text to check
            pipeline: Pipeline content to check against
        
        Returns:
            bool: True if rule is satisfied based on keyword matching
        """
        if not rule or not pipeline:
            return False
        
        rule_lower = rule.lower()
        pipeline_lower = pipeline.lower()
        
        # Extract key terms (3+ character words)
        key_terms = re.findall(r'\b[a-z]{3,}\b', rule_lower)
        
        if not key_terms:
            return False
        
        # Count how many key terms appear in pipeline
        matches = sum(1 for term in key_terms if term in pipeline_lower)
        match_ratio = matches / len(key_terms)
        
        # Use configurable threshold from settings
        threshold = self.settings.rule_satisfaction_keyword_threshold
        is_satisfied = match_ratio >= threshold
        
        logger.debug(
            f"[RULE] '{rule[:50]}...' match ratio: {match_ratio:.2%} "
            f"(threshold: {threshold:.0%}) = {'SATISFIED' if is_satisfied else 'VIOLATED'}"
        )
        
        return is_satisfied
    
    def _make_prediction(self, result: Dict, total_rules: int) -> Dict:
        """
        Make prediction based on RAW confidence 
        Outcome determined by confidence value
        
        Args:
            result: Evaluation results with satisfied/violated counts
            total_rules: Total number of rules evaluated
        
        Returns:
            Dict with 'outcome' and 'confidence' (0-100)
        """
        
        # Handle no rules case
        if total_rules == 0:
            if self.settings.prediction_unknown_on_no_rules:
                logger.warning("[PREDICTION] No rules found - returning UNKNOWN")
                return {'outcome': 'UNKNOWN', 'confidence': 50}
            else:
                logger.warning("[PREDICTION] No rules found - defaulting to PASS")
                return {'outcome': 'PASS', 'confidence': 70}
        
        satisfied = result.get('satisfied', 0)
        violated = result.get('violated', 0)
        
        # Calculate satisfaction ratio (internal only)
        satisfaction_ratio = satisfied / total_rules
        violation_ratio = violated / total_rules
        
        # Calculate confidence from satisfaction
        # More satisfied rules = higher confidence
        if violation_ratio < self.settings.prediction_high_risk_violation_threshold:
            # Low violations: boost confidence
            confidence = int(satisfaction_ratio * 100) + self.settings.prediction_pass_confidence_boost
            confidence = min(100, confidence)
        else:
            # Moderate to high violations: raw satisfaction
            confidence = int(satisfaction_ratio * 100)
        
        # Determine outcome based on confidence ranges
        if confidence <= self.settings.prediction_fail_max:
            outcome = 'FAIL'
        elif confidence <= self.settings.prediction_high_risk_max:
            outcome = 'HIGH_RISK'
        else:
            outcome = 'PASS'
        
        logger.info(f"[PREDICTION] {outcome} @ {confidence}%")
        
        return {
            'outcome': outcome,
            'confidence': confidence
        }

    
    def _build_comprehensive_response(self, pipeline, rules, result, docs, workspace_rules, prediction, ai_suggestions=None) -> str:
        """Build comprehensive analysis response with all findings"""
        
        response = f"""

ESTABLISHED RULES ANALYSIS:
{'='*70}
"""
        
        # Show first 50 rules with their status
        for i, rule in enumerate(rules[:50], 1):
            rule_text = rule.get('rule_text', '')[:100] if isinstance(rule, dict) else str(rule)[:100]
            status = "PASS" if self._rule_satisfied(rule_text, pipeline) else "FAIL"
            response += f"{i}. {rule_text}... - {status}\n"
        
        # Show workspace-specific findings
        if workspace_rules:
            response += f"\nREAL YOCTO WORKSPACE ANALYSIS:\n"
            response += f"Found {len(workspace_rules)} specific rules from actual Yocto files in your workspace\n"
            for r in workspace_rules[:10]:
                rule_text = r.get('rule_text', '') if isinstance(r, dict) else str(r)
                response += f"  * {rule_text}\n"
        
        # Summary statistics
        response += f"""
{'='*70}
ANALYSIS SUMMARY:
{'='*70}
APPLICABLE_RULES: {len(rules)}
SATISFIED_RULES: {result.get('satisfied', 0)}
VIOLATED_RULES: {result.get('violated', 0)}

{'='*70}
BUILD PREDICTION: {prediction['outcome']}
CONFIDENCE: {prediction['confidence']}%
{'='*70}

AI-GENERATED SUGGESTIONS TO IMPROVE CONFIDENCE:
"""
        
        # Add AI suggestions
        if ai_suggestions and len(ai_suggestions) > 0:
            logger.info(f"[BUILD] Adding {len(ai_suggestions)} AI suggestions")
            for i, suggestion in enumerate(ai_suggestions, 1):
                response += f"{i}. {suggestion}\n"
        else:
            logger.warning("[WARN] No AI suggestions generated")
            response += "No specific suggestions generated for this analysis.\n"
        
        # Add sources section
        response += "\n" + "="*70 + "\nSOURCES & REFERENCES:\n" + "="*70 + "\n"
        
        seen_sources = set()
        sources_list = []
        source_count = 1
        
        if docs and len(docs) > 0:
            logger.info(f"[BUILD] Processing {len(docs)} documents...")
            for doc in docs[:20]:
                try:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        # Confluence sources
                        confluence_url = doc.metadata.get('confluence_url', '')
                        if confluence_url:
                            source = doc.metadata.get('source', 'Confluence')
                            unique_key = f"confluence:{confluence_url}"
                            
                            if unique_key not in seen_sources:
                                seen_sources.add(unique_key)
                                page_name = source.split('/')[-1].replace('.md', '').replace('_', ' ') if '/' in source else 'Confluence'
                                sources_list.append(f"{source_count}. Confluence: {page_name} - {confluence_url}")
                                source_count += 1
                        
                        # Jenkins sources
                        source = doc.metadata.get('source', '')
                        if 'jenkins' in str(source).lower():
                            match = re.search(r'([a-zA-Z0-9_-]+)-(\d+)', str(source))
                            if match:
                                job, build = match.groups()
                                unique_key = f"jenkins:{job}:{build}"
                                
                                if unique_key not in seen_sources:
                                    seen_sources.add(unique_key)
                                    jenkins_url = f"{job}/{build}"
                                    sources_list.append(f"{source_count}. Jenkins: {job} #{build} - {jenkins_url}")
                                    source_count += 1
                except Exception as e:
                    logger.warning(f"[WARN] Error processing source: {e}")
        
        if sources_list:
            logger.info(f"[OK] Showing {len(sources_list)} UNIQUE sources")
            for source in sources_list:
                response += f"{source}\n"
        else:
            logger.warning("[WARN] No sources found")
            response += "No specific sources retrieved for this analysis.\n"
        
        response += "="*70 + "\n"
        
        return response
