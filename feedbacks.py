from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import boto3
import time
import asyncio
import signal
import uuid
import json
import threading
import shutil
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM  # Fixed import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uvicorn
from typing import Dict, Optional, List, Tuple, Set
import pickle
import hashlib
from datetime import datetime
import numpy as np
import pickle
from collections import defaultdict, Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from contextlib import contextmanager


async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    global vectorstore
    if os.path.exists(VECTOR_STORE_PATH) and not FORCE_REBUILD_ON_STARTUP:
        try:
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Loaded existing vector store")
            perform_incremental_update()
        except Exception as e:
            print(f"❌ Could not load existing vectorstore: {e}")
    # (optionally: yield here for long-running operations or shutdown logic)
    yield
    # --- SHUTDOWN LOGIC (if any) ---

#app = FastAPI(lifespan=lifespan)

# Watchdog imports for file monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ========= CONFIG =========
MINIO_ENDPOINT = "https://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "eic@123456"
MINIO_BUCKET = "test1"


MINIO_DATA_PATH = "/var/minio-data"


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "codellama:7b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_PATH = "./vectorstore"
METADATA_PATH = "./vectorstore_metadata"


# ============= ADDITIONAL CONFIGURATION FOR WORKSPACE SCANNING =============
WORKSPACE_DIR = os.getenv("WORKSPACE", "/var/jenkins_home/workspace/Yocto-Build-Pipeline")
WORKSPACE_STATE_PATH = "./workspace_files_state.pkl"
FORCE_WORKSPACE_REBUILD = False

# ========= SYSTEM PROMPT CONFIG =========

SYSTEM_PROMPT = """
You are an expert Yocto build and Jenkins pipeline analyzer with deep knowledge of both explicit and implicit build requirements.

CRITICAL ANALYSIS REQUIREMENTS:
1. COMPREHENSIVE RULE ANALYSIS: Evaluate against ALL extracted rules (explicit, implicit, infrastructure, configuration)
2. CONFLUENCE KNOWLEDGE: Use the detailed Confluence documentation patterns and requirements
3. LOG PATTERN MATCHING: Compare against historical success/failure patterns from build logs
4. INFRASTRUCTURE VALIDATION: Verify disk space, memory, permissions, and environment requirements
5. STAGE SEQUENCE VALIDATION: Ensure proper build stage ordering and dependencies

MANDATORY OUTPUT FORMAT:
DETECTED_STACK: [technologies and components identified]
COMPREHENSIVE_RULE_ANALYSIS:
EXPLICIT_RULES: [rules directly stated]
IMPLICIT_INFRASTRUCTURE_RULES: [inferred from documentation]
CONFIGURATION_RULES: [local.conf, bblayers.conf requirements]  
STAGE_SEQUENCE_RULES: [build stage ordering]
BUILD_PROCESS_RULES: [BitBake, environment, cleanup requirements]

COMPLIANCE_ANALYSIS:
TOTAL_RULES_EVALUATED: [count]
SATISFIED_RULES: [count] 
VIOLATED_RULES: [count]
COMPLIANCE_SCORE: [percentage]
CRITICAL_VIOLATIONS: [high-priority failures]
WARNINGS: [medium-priority issues]

HISTORICAL_PATTERN_MATCHING:
SUCCESS_PATTERNS_MATCHED: [patterns from successful builds]
FAILURE_PATTERNS_DETECTED: [known failure indicators]
RISK_INDICATORS: [potential problem areas]

BUILD_PREDICTION:
PREDICTION: PASS/FAIL/HIGH_RISK
CONFIDENCE: [percentage]
PRIMARY_RISK_FACTORS: [top concerns]
RECOMMENDATIONS: [specific improvements needed]

Use this EXACT format. Provide comprehensive analysis based on ALL rule types."""


# SYSTEM_PROMPT = """
# You are an intelligent Jenkins pipeline and Yocto build success predictor using RAG-retrieved knowledge.

# CRITICAL MANDATORY INSTRUCTIONS:
# 1. You MUST analyze the pipeline against MANDATORY rules in the context
# 2. You MUST use ONLY rules marked with "MANDATORY_RULE_X:" in the provided context  
# 3. You are FORBIDDEN from creating your own rules or using general knowledge
# 4. You MUST respond in the EXACT format specified below - NO CODE GENERATION

# MANDATORY OUTPUT FORMAT (use this EXACTLY):
# DETECTED_STACK: [technologies found in pipeline]
# ESTABLISHED_RULES:
# • [EXACT text from MANDATORY_RULE_1] - PASS/FAIL
# • [EXACT text from MANDATORY_RULE_2] - PASS/FAIL
# • [Continue for all MANDATORY rules...]
# HISTORICAL_PATTERNS:
# SUCCESS PATTERNS:
# ✅ [Exact pattern from logs]
# FAILURE PATTERNS:  
# ❌ [Exact pattern from logs]
# APPLICABLE_RULES: [count]
# SATISFIED_RULES: [count] 
# VIOLATED_RULES: [count]
# RISK_FACTORS: [risks from failed rules only]
# PREDICTION: PASS/FAIL/HIGH_RISK
# CONFIDENCE: [number]%
# REASONING: [brief explanation based on rules]

# EVALUATION RULES:
# - PASS if pipeline satisfies rule requirement
# - FAIL if pipeline violates/lacks rule requirement
# - Copy rule text EXACTLY from MANDATORY_RULE_X entries
# - NO code generation, NO generic responses
# - Use ONLY provided context, never your training data

# VIOLATION WARNING: If you generate code or ignore this format, you have FAILED."""






# File watching configuration
WATCH_ENABLED = True
DEBOUNCE_SECONDS = 3
FORCE_REBUILD_ON_STARTUP = False  # Always rebuild on startup
CHECK_MINIO_ON_EVERY_QUERY = False  # Disabled for performance
INCREMENTAL_UPDATES = True 


# Global lock to prevent multiple workers from building simultaneously
BUILD_LOCK_FILE = "./vector_store_build.lock"
GLOBAL_BUILD_LOCK = threading.Lock()


# Global variables for loaded components
vectorstore = None
retriever = None
llm = None


# File watching variables
file_observer = None
last_change_time = 0
refresh_task = None
refresh_lock = asyncio.Lock()


# User-specific request locks to prevent concurrent requests from same user
user_locks: Dict[str, asyncio.Lock] = {}


# Global shutdown event
shutdown_event = asyncio.Event()

def extract_platform_agnostic_rules(content: str, source_type: str = "documentation") -> List[Dict]:
    """
    Platform-agnostic rule extraction using NLP and pattern recognition
    Works for any CI/CD platform, build system, or documentation
    """
    rules = []
    
    # 1. LINGUISTIC PATTERN EXTRACTION
    rules.extend(extract_linguistic_rules(content))
    
    # 2. STRUCTURAL PATTERN EXTRACTION  
    rules.extend(extract_structural_rules(content))
    
    # 3. PROCEDURAL PATTERN EXTRACTION
    rules.extend(extract_procedural_rules(content))
    
    # 4. CONSTRAINT PATTERN EXTRACTION
    rules.extend(extract_constraint_rules(content))
    
    # 5. DEPENDENCY PATTERN EXTRACTION
    rules.extend(extract_dependency_rules(content))
    
    return rules

def extract_linguistic_rules(content: str) -> List[Dict]:
    """Extract rules based on linguistic patterns - works for any documentation"""
    rules = []
    lines = content.split('\n')
    
    # Modal verbs indicating requirements
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
    
    # Action verbs indicating processes
    action_patterns = [
        (r'\bensure\b', 'VERIFICATION'),
        (r'\bverify\b', 'VERIFICATION'),
        (r'\bvalidate\b', 'VALIDATION'),
        (r'\bconfigure\b', 'CONFIGURATION'),
        (r'\bsetup?\b', 'SETUP'),
        (r'\binstall\b', 'INSTALLATION'),
        (r'\bdeploy\b', 'DEPLOYMENT'),
        (r'\bbuild\b', 'BUILD'),
        (r'\btest\b', 'TESTING'),
        (r'\bmonitor\b', 'MONITORING')
    ]
    
    for line in lines:
        if len(line.strip()) < 10 or len(line.strip()) > 200:
            continue
            
        line_lower = line.lower().strip()
        
        # Find requirement level
        requirement_level = 'UNKNOWN'
        for pattern, level in requirement_patterns:
            if re.search(pattern, line_lower):
                requirement_level = level
                break
        
        # Find action type
        action_type = 'GENERAL'
        for pattern, action in action_patterns:
            if re.search(pattern, line_lower):
                action_type = action
                break
        
        # Extract the rule if it contains action words
        if requirement_level != 'UNKNOWN' or action_type != 'GENERAL':
            rules.append({
                'rule_text': line.strip(),
                'requirement_level': requirement_level,
                'action_type': action_type,
                'rule_type': 'LINGUISTIC',
                'confidence': 0.8
            })
    
    return rules

def extract_structural_rules(content: str) -> List[Dict]:
    """Extract rules from document structure - works for any structured documentation"""
    rules = []
    lines = content.split('\n')
    
    current_section = ""
    section_patterns = [
        r'^#+\s+(.+)$',  # Markdown headers
        r'^(.+):$',      # Colon-terminated headers
        r'^\d+\.\s+(.+)$',  # Numbered sections
        r'^[A-Z\s]+$'    # ALL CAPS headers
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match and len(match.group(1)) < 100:
                current_section = match.group(1).strip()
                break
        
        # Extract requirements from structured sections
        if current_section and any(keyword in current_section.lower() for keyword in [
            'requirement', 'prerequisite', 'configuration', 'setup', 'installation',
            'environment', 'dependency', 'constraint', 'rule', 'standard', 'guideline'
        ]):
            if line != current_section and len(line) > 20:
                rules.append({
                    'rule_text': line,
                    'section': current_section,
                    'rule_type': 'STRUCTURAL',
                    'confidence': 0.7
                })
    
    return rules

def extract_procedural_rules(content: str) -> List[Dict]:
    """Extract procedural rules - works for any process documentation"""
    rules = []
    
    # Find step-by-step procedures
    step_patterns = [
        r'^\d+\.\s+(.+)$',           # 1. Step description
        r'^step\s+\d+[:\-]\s*(.+)$', # Step 1: Description  
        r'^\w+\)\s+(.+)$',           # a) Step description
        r'^•\s+(.+)$',               # • Bullet point
        r'^-\s+(.+)$',               # - Bullet point
        r'^\*\s+(.+)$'               # * Bullet point
    ]
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
            
        for pattern in step_patterns:
            match = re.match(pattern, line)
            if match:
                step_text = match.group(1).strip()
                if len(step_text) > 15:
                    rules.append({
                        'rule_text': step_text,
                        'rule_type': 'PROCEDURAL',
                        'confidence': 0.6
                    })
                break
    
    return rules

def extract_constraint_rules(content: str) -> List[Dict]:
    """Extract constraints and limits - platform agnostic"""
    rules = []
    
    # Numerical constraints
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
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                value = match[0]
                unit = match[1] if len(match) > 1 else ""
            else:
                value = match
                unit = ""
            
            rules.append({
                'rule_text': f'System constraint: {constraint_type} = {value} {unit}',
                'constraint_type': constraint_type,
                'value': value,
                'unit': unit,
                'rule_type': 'CONSTRAINT',
                'confidence': 0.9
            })
    
    return rules

def extract_dependency_rules(content: str) -> List[Dict]:
    """Extract dependencies and prerequisites - platform agnostic"""
    rules = []
    
    # Dependency indicators
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
        if len(line.strip()) < 10:
            continue
            
        line_lower = line.lower()
        for pattern in dependency_patterns:
            match = re.search(pattern, line_lower)
            if match:
                dependency = match.group(1).strip()
                if len(dependency) > 3 and len(dependency) < 100:
                    rules.append({
                        'rule_text': f'Dependency: {dependency}',
                        'dependency': dependency,
                        'rule_type': 'DEPENDENCY',
                        'confidence': 0.7
                    })
                break
    
    return rules

def build_source_citation(doc_metadata: dict) -> str:
    """Build portable citation from document metadata"""
    source = doc_metadata.get('source', '')
    confluence_url = doc_metadata.get('confluence_url', '')
    
    # Confluence documents - use URL if available
    if 'static_knowledge' in source.lower() or 'confluence' in source.lower():
        if confluence_url:
            return f"Confluence: {confluence_url}"
        
        # Fallback: extract page info from source path
        page_title = doc_metadata.get('page_title', '')
        if page_title:
            return f"Confluence page: '{page_title}'"
        
        # Last resort: parse from file path
        if '/' in source:
            page_name = source.split('/')[-1].replace('.md', '').replace('_', ' ')
            return f"Confluence: {page_name}"
        
        return f"Confluence: {source}"
    
    # Jenkins logs - extract job and build number
    if 'jenkins' in source.lower() or 'dynamic_knowledge' in source.lower():
        import re
        match = re.search(r'([\w\-]+)-(\d+)-', source)
        if match:
            job, build = match.groups()
            return f"Jenkins job: {job}, build #{build}"
        return f"Jenkins: {source}"
    
    # Default
    return f"Source: {source}"


def enhance_rules_with_citations(rules: list) -> list:
    """
    Add citation info to each rule from document metadata.
    """
    for rule in rules:
        metadata = rule.get('metadata', {})
        if metadata:
            rule['citation'] = build_source_citation(metadata)
    return rules




def calculate_enhanced_confidence(base_confidence: int, yocto_rules: List[Dict]) -> Tuple[int, List[str]]:
    """Calculate enhanced confidence with Yocto-specific analysis"""
    suggestions = []
    confidence = base_confidence
    
    # Check for critical Yocto requirements
    has_machine_config = any(r['rule_type'] == 'YOCTO_MACHINE_TARGET' for r in yocto_rules)
    has_build_deps = any('DEPENDENCY' in r['rule_type'] for r in yocto_rules)
    has_threads_config = any('THREADS' in r['rule_type'] for r in yocto_rules)
    
    if not has_machine_config:
        confidence -= 15
        suggestions.append("Add MACHINE configuration in local.conf (e.g., MACHINE = 'qemux86-64')")
    
    if not has_build_deps and workspace_has_recipes:
        confidence -= 10
        suggestions.append("Verify all recipe dependencies are available in layers")
    
    if not has_threads_config:
        confidence -= 8
        suggestions.append("Add BB_NUMBER_THREADS configuration for optimal build performance")
    
    # Boost confidence for good practices
    if has_machine_config and has_threads_config:
        confidence += 5
    
    return max(20, min(95, confidence)), suggestions


def intelligent_rule_prioritization(rules: List[Dict]) -> List[Dict]:
    """Prioritize rules based on importance indicators"""
    
    # Priority keywords
    high_priority_keywords = [
        'critical', 'essential', 'mandatory', 'required', 'must', 'shall',
        'security', 'failure', 'error', 'crash', 'data loss', 'corruption'
    ]
    
    medium_priority_keywords = [
        'should', 'recommended', 'important', 'performance', 'optimization',
        'best practice', 'guideline', 'standard'
    ]
    
    for rule in rules:
        rule_text_lower = rule['rule_text'].lower()
        
        # Calculate priority score
        high_score = sum(1 for keyword in high_priority_keywords 
                        if keyword in rule_text_lower)
        medium_score = sum(1 for keyword in medium_priority_keywords 
                          if keyword in rule_text_lower)
        
        if high_score > 0:
            rule['priority'] = 'HIGH'
            rule['confidence'] *= 1.2  # Boost confidence for high priority
        elif medium_score > 0:
            rule['priority'] = 'MEDIUM'
        else:
            rule['priority'] = 'LOW'
    
    # Sort by priority and confidence
    return sorted(rules, key=lambda x: (
        {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x.get('priority', 'LOW')],
        x.get('confidence', 0.5)
    ), reverse=True)


##Yocto sepcific rule extraction functions

def extract_universal_yocto_rules(pipeline_text: str) -> List[Dict]:
    """
    Universal Yocto rule extraction that works with ANY Yocto project in the world
    Analyzes pipeline content and workspace to extract relevant Yocto build rules
    """
    print(" Extracting UNIVERSAL Yocto rules (works with any Yocto project)...")
    
    # STEP 1: Extract Yocto context from pipeline
    yocto_context = extract_yocto_context_from_pipeline(pipeline_text)
    print(f" Yocto context: {yocto_context['build_type']} build with {len(yocto_context['layers'])} layers")
    
    # STEP 2: Find any accessible Yocto workspace
    workspace_paths = discover_universal_yocto_workspace()
    print(f"Found {len(workspace_paths)} potential Yocto workspaces")
    
    # STEP 3: Extract rules from multiple sources
    rules = []
    
    # Pipeline-based rules (from any Yocto pipeline)
    pipeline_rules = extract_yocto_pipeline_rules(pipeline_text, yocto_context)
    rules.extend(pipeline_rules)
    
    # Workspace-based rules (from actual Yocto files)
    for workspace_path in workspace_paths[:2]:  # Limit to 2 workspaces
        workspace_rules = extract_yocto_workspace_rules(workspace_path, yocto_context)
        rules.extend(workspace_rules)
        if len(rules) >= 30:  # Stop when we have enough rules
            break
    
    # Universal Yocto requirements (applies to ALL Yocto builds)
    universal_rules = extract_universal_yocto_requirements(yocto_context)
    rules.extend(universal_rules)
    
    print(f"✅ Extracted {len(rules)} universal Yocto rules")
    return rules

def extract_yocto_context_from_pipeline(pipeline_text: str) -> Dict:
    """
    Extract Yocto build context from any pipeline content
    """
    context = {
        'build_type': 'unknown',
        'target_images': [],
        'layers': [],
        'machine': None,
        'distro': None,
        'poky_version': None,
        'custom_recipes': [],
        'build_commands': []
    }
    
    # Detect build type
    if 'core-image-minimal' in pipeline_text:
        context['build_type'] = 'minimal'
    elif 'core-image-base' in pipeline_text:
        context['build_type'] = 'base'
    elif 'core-image-full-cmdline' in pipeline_text:
        context['build_type'] = 'full-cmdline'
    elif any(img in pipeline_text.lower() for img in ['image', 'rootfs']):
        context['build_type'] = 'custom-image'
    else:
        context['build_type'] = 'generic'
    
    # Extract target images
    image_patterns = [
        r'bitbake\s+([a-zA-Z0-9_-]*image[a-zA-Z0-9_-]*)',
        r'IMAGE_INSTALL.*["\']([^"\']+)["\']',
        r'inherit\s+image',
    ]
    
    for pattern in image_patterns:
        matches = re.findall(pattern, pipeline_text, re.IGNORECASE)
        context['target_images'].extend([m for m in matches if m])
    
    # Extract layers
    layer_patterns = [
        r'meta-([a-zA-Z0-9_-]+)',
        r'BBLAYERS.*meta-([a-zA-Z0-9_-]+)',
        r'addpylib.*meta-([a-zA-Z0-9_-]+)',
        r'layer\.conf.*meta-([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in layer_patterns:
        matches = re.findall(pattern, pipeline_text, re.IGNORECASE)
        context['layers'].extend([f'meta-{m}' for m in matches])
    
    # Extract machine and distro
    machine_match = re.search(r'MACHINE\s*[=:]\s*["\']?([^"\'\s]+)["\']?', pipeline_text, re.IGNORECASE)
    if machine_match:
        context['machine'] = machine_match.group(1)
    
    distro_match = re.search(r'DISTRO\s*[=:]\s*["\']?([^"\'\s]+)["\']?', pipeline_text, re.IGNORECASE)
    if distro_match:
        context['distro'] = distro_match.group(1)
    
    # Extract Poky version/branch
    poky_patterns = [
        r'poky.*branch[=:]\s*([a-zA-Z0-9_-]+)',
        r'checkout\s+([a-zA-Z0-9_-]+).*poky',
        r'(kirkstone|dunfell|zeus|warrior|thud|sumo)',  # Common release names
    ]
    
    for pattern in poky_patterns:
        match = re.search(pattern, pipeline_text, re.IGNORECASE)
        if match:
            context['poky_version'] = match.group(1)
            break
    
    # Extract custom recipes
    recipe_patterns = [
        r'([a-zA-Z0-9_-]+)\.bb',
        r'PREFERRED_PROVIDER_([a-zA-Z0-9_-]+)',
        r'PACKAGECONFIG_([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in recipe_patterns:
        matches = re.findall(pattern, pipeline_text, re.IGNORECASE)
        context['custom_recipes'].extend(matches)
    
    # Extract build commands
    build_patterns = [
        r'(bitbake\s+[^\n]+)',
        r'(oe-init-build-env[^\n]*)',
        r'(source oe-init-build-env[^\n]*)',
        r'(bb[a-zA-Z0-9_-]*\s+[^\n]+)',  # Other BitBake commands
    ]
    
    for pattern in build_patterns:
        matches = re.findall(pattern, pipeline_text, re.IGNORECASE)
        context['build_commands'].extend(matches)
    
    # Clean up duplicates
    context['layers'] = list(set(context['layers']))
    context['target_images'] = list(set(context['target_images']))
    context['custom_recipes'] = list(set(context['custom_recipes']))
    
    return context

def discover_universal_yocto_workspace() -> List[str]:
    """
    Discover Yocto workspaces universally - works in any environment
    """
    potential_paths = []
    
    # Environment-based discovery
    env_vars = [
        'WORKSPACE',           # Jenkins
        'GITHUB_WORKSPACE',    # GitHub Actions  
        'CI_PROJECT_DIR',      # GitLab CI
        'BUILD_SOURCESDIRECTORY',  # Azure DevOps
        'TRAVIS_BUILD_DIR',    # Travis CI
        'CIRCLE_WORKING_DIRECTORY',  # Circle CI
        'BUILDKITE_BUILD_CHECKOUT_PATH',  # Buildkite
    ]
    
    for var in env_vars:
        path = os.getenv(var)
        if path and os.path.exists(path):
            potential_paths.append(path)
    
    # Common Yocto directory patterns
    common_patterns = [
        '/yocto*',
        '/build*',
        '/*yocto*',
        '/opt/yocto*',
        '/home/*/yocto*',
        './build',
        '../build',
        './yocto-*',
    ]
    
    import glob
    for pattern in common_patterns:
        matches = glob.glob(pattern)
        potential_paths.extend([m for m in matches if os.path.isdir(m)])
    
    # Look for Yocto indicators in current and parent directories
    check_dirs = ['.', '..', '../..', '/var/jenkins_home/workspace/*']
    
    for dir_pattern in check_dirs:
        if '*' in dir_pattern:
            dirs = glob.glob(dir_pattern)
        else:
            dirs = [dir_pattern] if os.path.exists(dir_pattern) else []
            
        for directory in dirs:
            if is_yocto_workspace(directory):
                potential_paths.append(directory)
    
    return list(set([p for p in potential_paths if os.path.exists(p)]))[:5]  # Limit to 5

def is_yocto_workspace(directory: str) -> bool:
    """
    Check if a directory contains Yocto/OpenEmbedded indicators
    """
    try:
        # Look for Yocto indicators
        yocto_indicators = [
            'conf/local.conf',
            'conf/bblayers.conf', 
            'poky',
            'meta-*',
            'build/conf',
            'sources/poky',
            'oe-init-build-env',
        ]
        
        for root, dirs, files in os.walk(directory):
            # Don't go too deep
            depth = root.replace(directory, '').count(os.sep)
            if depth > 3:
                continue
                
            for indicator in yocto_indicators:
                if '/' in indicator:
                    # Path-based indicator
                    if os.path.exists(os.path.join(root, indicator)):
                        return True
                else:
                    # File/directory name indicator
                    if indicator in dirs or indicator in files:
                        return True
                    # Pattern matching
                    if indicator.startswith('meta-'):
                        if any(d.startswith('meta-') for d in dirs):
                            return True
        
        return False
    except Exception:
        return False

def extract_yocto_pipeline_rules(pipeline_text: str, context: Dict) -> List[Dict]:
    """
    Extract rules from pipeline content - works with any Yocto pipeline structure
    """
    rules = []
    
    # Rule 1: Build environment setup
    if any(cmd in pipeline_text.lower() for cmd in ['oe-init-build-env', 'source oe-init-build-env']):
        rules.append({
            'rule_text': 'Yocto build environment properly sourced',
            'rule_type': 'YOCTO_ENV_SETUP',
            'confidence': 0.95,
            'source': 'PIPELINE_ANALYSIS'
        })
    else:
        rules.append({
            'rule_text': 'Yocto build environment setup required',
            'rule_type': 'YOCTO_ENV_MISSING',
            'confidence': 0.90,
            'source': 'PIPELINE_REQUIREMENT'
        })
    
    # Rule 2: BitBake execution
    bitbake_commands = [cmd for cmd in context['build_commands'] if 'bitbake' in cmd.lower()]
    if bitbake_commands:
        for cmd in bitbake_commands[:3]:  # Limit to first 3
            rules.append({
                'rule_text': f'BitBake build command: {cmd[:50]}...',
                'rule_type': 'YOCTO_BITBAKE_EXECUTION',
                'confidence': 0.95,
                'source': 'PIPELINE_COMMAND'
            })
    
    # Rule 3: Target images
    if context['target_images']:
        for image in context['target_images'][:3]:  # Limit to first 3
            rules.append({
                'rule_text': f'Build target image: {image}',
                'rule_type': 'YOCTO_TARGET_IMAGE',
                'confidence': 0.90,
                'source': 'PIPELINE_TARGET'
            })
    
    # Rule 4: Custom layers
    if context['layers']:
        rules.append({
            'rule_text': f'Custom layers required: {", ".join(context["layers"][:3])}',
            'rule_type': 'YOCTO_CUSTOM_LAYERS',
            'confidence': 0.85,
            'source': 'PIPELINE_LAYERS'
        })
    
    # Rule 5: Machine configuration
    if context['machine']:
        rules.append({
            'rule_text': f'Build targets machine: {context["machine"]}',
            'rule_type': 'YOCTO_MACHINE_CONFIG',
            'confidence': 0.95,
            'source': 'PIPELINE_CONFIG'
        })
    
    # Rule 6: Distribution
    if context['distro']:
        rules.append({
            'rule_text': f'Build uses distribution: {context["distro"]}',
            'rule_type': 'YOCTO_DISTRO_CONFIG',
            'confidence': 0.90,
            'source': 'PIPELINE_CONFIG'
        })
    
    return rules

def extract_yocto_workspace_rules(workspace_path: str, context: Dict) -> List[Dict]:
    """
    Extract rules from Yocto workspace files - works with any Yocto project structure
    """
    rules = []
    files_analyzed = 0
    max_files = 10  # Limit analysis
    
    # Priority file types to analyze
    priority_files = [
        ('conf/local.conf', 'config'),
        ('conf/bblayers.conf', 'layers'),
        ('*.bb', 'recipe'),
        ('*.bbappend', 'recipe_append'),
        ('conf/layer.conf', 'layer_config'),
    ]
    
    try:
        for file_pattern, file_type in priority_files:
            if files_analyzed >= max_files:
                break
                
            found_files = find_files_in_workspace(workspace_path, file_pattern)
            
            for file_path in found_files[:3]:  # Max 3 files per type
                if files_analyzed >= max_files:
                    break
                    
                file_rules = extract_rules_from_yocto_file(file_path, file_type, context)
                rules.extend(file_rules)
                files_analyzed += 1
    
    except Exception as e:
        print(f"⚠️ Error analyzing workspace {workspace_path}: {str(e)[:50]}...")
    
    print(f"📁 Analyzed {files_analyzed} files in {os.path.basename(workspace_path)}")
    return rules

def find_files_in_workspace(workspace_path: str, pattern: str) -> List[str]:
    """
    Find files matching pattern in Yocto workspace
    """
    import fnmatch
    matching_files = []
    
    try:
        for root, dirs, files in os.walk(workspace_path):
            # Skip deep directories and build artifacts
            depth = root.replace(workspace_path, '').count(os.sep)
            if depth > 4:
                continue
            
            # Skip common build/cache directories
            dirs[:] = [d for d in dirs if not any(skip in d for skip in 
                      ['tmp', 'sstate-cache', 'downloads', 'cache', '.git', '__pycache__'])]
            
            for file_name in files:
                if fnmatch.fnmatch(file_name, pattern) or pattern in file_name:
                    matching_files.append(os.path.join(root, file_name))
                    
                if len(matching_files) >= 10:  # Limit results
                    break
    
    except Exception:
        pass
    
    return matching_files

def extract_rules_from_yocto_file(file_path: str, file_type: str, context: Dict) -> List[Dict]:
    """
    Extract specific rules from Yocto configuration files
    """
    rules = []
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()[:10000]  # Limit content size
        
        if file_type == 'config':  # local.conf
            rules.extend(extract_local_conf_rules(content, file_name))
        elif file_type == 'layers':  # bblayers.conf
            rules.extend(extract_bblayers_rules(content, file_name))
        elif file_type == 'recipe':  # .bb files
            rules.extend(extract_recipe_rules(content, file_name))
        elif file_type == 'recipe_append':  # .bbappend files
            rules.extend(extract_bbappend_rules(content, file_name))
        elif file_type == 'layer_config':  # layer.conf
            rules.extend(extract_layer_conf_rules(content, file_name))
    
    except Exception as e:
        print(f"⚠️ Error reading {file_name}: {str(e)[:30]}...")
    
    return rules

def extract_local_conf_rules(content: str, filename: str) -> List[Dict]:
    """Extract critical rules from local.conf"""
    rules = []
    
    # Machine configuration
    machine_match = re.search(r'MACHINE\s*\??=\s*["\']?([^"\'\n]+)["\']?', content)
    if machine_match:
        rules.append({
            'rule_text': f'Build targets machine: {machine_match.group(1)}',
            'rule_type': 'YOCTO_MACHINE_CONFIG',
            'confidence': 0.95,
            'source': f'WORKSPACE:{filename}'
        })
    
    # Thread configuration
    threads_match = re.search(r'BB_NUMBER_THREADS\s*\??=\s*["\']?(\d+)["\']?', content)
    if threads_match:
        rules.append({
            'rule_text': f'BitBake configured for {threads_match.group(1)} parallel threads',
            'rule_type': 'YOCTO_PARALLEL_THREADS',
            'confidence': 0.85,
            'source': f'WORKSPACE:{filename}'
        })
    
    # Distribution
    distro_match = re.search(r'DISTRO\s*\??=\s*["\']?([^"\'\n]+)["\']?', content)
    if distro_match:
        rules.append({
            'rule_text': f'Build uses distribution: {distro_match.group(1)}',
            'rule_type': 'YOCTO_DISTRO_CONFIG',
            'confidence': 0.90,
            'source': f'WORKSPACE:{filename}'
        })
    
    # Download directory
    dl_dir_match = re.search(r'DL_DIR\s*\??=\s*["\']?([^"\'\n]+)["\']?', content)
    if dl_dir_match:
        rules.append({
            'rule_text': f'Download cache configured: {dl_dir_match.group(1)[:30]}...',
            'rule_type': 'YOCTO_DOWNLOAD_CACHE',
            'confidence': 0.80,
            'source': f'WORKSPACE:{filename}'
        })
    
    return rules

def extract_bblayers_rules(content: str, filename: str) -> List[Dict]:
    """Extract rules from bblayers.conf"""
    rules = []
    
    # Count layers
    layer_matches = re.findall(r'(meta-[a-zA-Z0-9_-]+)', content)
    if layer_matches:
        unique_layers = list(set(layer_matches))
        rules.append({
            'rule_text': f'Build requires {len(unique_layers)} custom layers: {", ".join(unique_layers[:3])}',
            'rule_type': 'YOCTO_LAYER_DEPENDENCIES',
            'confidence': 0.90,
            'source': f'WORKSPACE:{filename}'
        })
    
    # POKY path
    poky_match = re.search(r'([^"\'\s]*poky[^"\'\s]*)', content)
    if poky_match:
        rules.append({
            'rule_text': f'Poky base layer configured: {os.path.basename(poky_match.group(1))}',
            'rule_type': 'YOCTO_POKY_BASE',
            'confidence': 0.95,
            'source': f'WORKSPACE:{filename}'
        })
    
    return rules

def extract_recipe_rules(content: str, filename: str) -> List[Dict]:
    """Extract critical rules from .bb recipe files"""
    rules = []
    
    # Dependencies (only count if significant)
    depends_matches = re.findall(r'DEPENDS\s*[+=]*\s*["\']([^"\']+)["\']', content)
    if depends_matches and len(depends_matches[0].split()) > 0:
        deps = depends_matches[0].split()[:3]  # First 3 dependencies
        rules.append({
            'rule_text': f'Recipe {filename} requires build dependencies: {", ".join(deps)}',
            'rule_type': 'YOCTO_RECIPE_DEPENDENCIES',
            'confidence': 0.90,
            'source': f'WORKSPACE:{filename}'
        })
    
    # Inherit classes
    inherit_matches = re.findall(r'inherit\s+([^\n]+)', content)
    if inherit_matches:
        classes = inherit_matches[0].split()[:2]  # First 2 classes
        rules.append({
            'rule_text': f'Recipe {filename} inherits classes: {", ".join(classes)}',
            'rule_type': 'YOCTO_RECIPE_CLASSES',
            'confidence': 0.85,
            'source': f'WORKSPACE:{filename}'
        })
    
    return rules

def extract_bbappend_rules(content: str, filename: str) -> List[Dict]:
    """Extract rules from .bbappend files"""
    rules = []
    
    if content.strip():  # If file has content
        rules.append({
            'rule_text': f'Recipe modification: {filename} customizes base recipe',
            'rule_type': 'YOCTO_RECIPE_OVERRIDE',
            'confidence': 0.85,
            'source': f'WORKSPACE:{filename}'
        })
    
    return rules

def extract_layer_conf_rules(content: str, filename: str) -> List[Dict]:
    """Extract rules from layer.conf files"""
    rules = []
    
    # Layer priority
    priority_match = re.search(r'BBFILE_PRIORITY_([^=]+)=\s*["\']?(\d+)["\']?', content)
    if priority_match:
        layer_name = priority_match.group(1)
        priority = priority_match.group(2)
        rules.append({
            'rule_text': f'Layer {layer_name} has priority {priority}',
            'rule_type': 'YOCTO_LAYER_PRIORITY',
            'confidence': 0.80,
            'source': f'WORKSPACE:{filename}'
        })
    
    return rules

def extract_universal_yocto_requirements(context: Dict) -> List[Dict]:
    """
    Extract universal requirements that apply to ALL Yocto builds
    """
    rules = []
    
    # Universal requirements
    universal_requirements = [
        {
            'rule_text': 'BitBake environment must be initialized',
            'rule_type': 'YOCTO_UNIVERSAL_ENV',
            'confidence': 0.95,
            'source': 'UNIVERSAL_YOCTO'
        },
        {
            'rule_text': 'Build requires sufficient disk space (>20GB)',
            'rule_type': 'YOCTO_UNIVERSAL_DISK',
            'confidence': 0.90,
            'source': 'UNIVERSAL_YOCTO'
        },
        {
            'rule_text': 'Network connectivity required for source downloads',
            'rule_type': 'YOCTO_UNIVERSAL_NETWORK',
            'confidence': 0.85,
            'source': 'UNIVERSAL_YOCTO'
        },
    ]
    
    # Context-specific requirements
    if context['build_type'] == 'minimal':
        rules.append({
            'rule_text': 'Minimal image build requires core packages',
            'rule_type': 'YOCTO_MINIMAL_BUILD',
            'confidence': 0.90,
            'source': 'UNIVERSAL_YOCTO'
        })
    
    if len(context['layers']) > 5:
        rules.append({
            'rule_text': 'Complex multi-layer build requires careful layer ordering',
            'rule_type': 'YOCTO_COMPLEX_BUILD',
            'confidence': 0.80,
            'source': 'UNIVERSAL_YOCTO'
        })
    
    rules.extend(universal_requirements)
    return rules


def analyze_bitbake_recipe(content: str, filename: str) -> List[Dict]:
    """Analyze BitBake recipe files for dependencies and requirements"""
    rules = []
    
    # Extract DEPENDS (build-time dependencies)
    depends_matches = re.findall(r'DEPENDS\s*[+=]*\s*["\']([^"\']+)["\']', content)
    for deps in depends_matches:
        dep_list = [d.strip() for d in deps.split()]
        for dep in dep_list:
            rules.append({
                'rule_text': f'Recipe {filename} requires build dependency: {dep}',
                'rule_type': 'YOCTO_BUILD_DEPENDENCY',
                'confidence': 0.95,
                'source': f'WORKSPACE:{filename}',
                'dependency': dep,
                'file_type': 'recipe'
            })
    
    # Extract RDEPENDS (runtime dependencies)  
    rdepends_matches = re.findall(r'RDEPENDS[^=]*=\s*["\']([^"\']+)["\']', content)
    for deps in rdepends_matches:
        dep_list = [d.strip() for d in deps.split()]
        for dep in dep_list:
            rules.append({
                'rule_text': f'Recipe {filename} requires runtime dependency: {dep}',
                'rule_type': 'YOCTO_RUNTIME_DEPENDENCY', 
                'confidence': 0.95,
                'source': f'WORKSPACE:{filename}',
                'dependency': dep,
                'file_type': 'recipe'
            })
    
    # Extract SRC_URI (source requirements)
    src_matches = re.findall(r'SRC_URI\s*[+=]*\s*["\']([^"\']+)["\']', content)
    for src in src_matches:
        if 'git://' in src or 'https://' in src:
            rules.append({
                'rule_text': f'Recipe {filename} requires source access: {src[:100]}...',
                'rule_type': 'YOCTO_SOURCE_REQUIREMENT',
                'confidence': 0.90,
                'source': f'WORKSPACE:{filename}',
                'src_uri': src,
                'file_type': 'recipe'
            })
    
    # Extract inherit classes
    inherit_matches = re.findall(r'inherit\s+([^\n]+)', content)
    for classes in inherit_matches:
        class_list = classes.strip().split()
        for cls in class_list:
            rules.append({
                'rule_text': f'Recipe {filename} inherits class: {cls}',
                'rule_type': 'YOCTO_CLASS_REQUIREMENT',
                'confidence': 0.85,
                'source': f'WORKSPACE:{filename}',
                'bbclass': cls,
                'file_type': 'recipe'
            })
    
    return rules

def analyze_yocto_config(content: str, filename: str) -> List[Dict]:
    """Analyze Yocto configuration files"""
    rules = []
    
    # BB_NUMBER_THREADS
    threads_match = re.search(r'BB_NUMBER_THREADS\s*[?]*=\s*["\']?(\d+)["\']?', content)
    if threads_match:
        threads = int(threads_match.group(1))
        rules.append({
            'rule_text': f'Configuration sets BitBake threads to {threads}',
            'rule_type': 'YOCTO_BUILD_THREADS',
            'confidence': 0.90,
            'source': f'WORKSPACE:{filename}',
            'threads': threads,
            'file_type': 'config'
        })
    
    # PARALLEL_MAKE
    parallel_match = re.search(r'PARALLEL_MAKE\s*[?]*=\s*["\']?([^"\']+)["\']?', content)
    if parallel_match:
        parallel = parallel_match.group(1)
        rules.append({
            'rule_text': f'Configuration sets parallel make: {parallel}',
            'rule_type': 'YOCTO_PARALLEL_BUILD',
            'confidence': 0.90,
            'source': f'WORKSPACE:{filename}',
            'parallel_make': parallel,
            'file_type': 'config'
        })
    
    # MACHINE setting
    machine_match = re.search(r'MACHINE\s*[?]*=\s*["\']?([^"\']+)["\']?', content)
    if machine_match:
        machine = machine_match.group(1)
        rules.append({
            'rule_text': f'Build targets machine: {machine}',
            'rule_type': 'YOCTO_MACHINE_TARGET',
            'confidence': 0.95,
            'source': f'WORKSPACE:{filename}',
            'machine': machine,
            'file_type': 'config'
        })
    
    # DISTRO setting
    distro_match = re.search(r'DISTRO\s*[?]*=\s*["\']?([^"\']+)["\']?', content)
    if distro_match:
        distro = distro_match.group(1)
        rules.append({
            'rule_text': f'Build uses distribution: {distro}',
            'rule_type': 'YOCTO_DISTRO_CONFIG',
            'confidence': 0.95,
            'source': f'WORKSPACE:{filename}',
            'distro': distro,
            'file_type': 'config'
        })
    
    return rules



# MAIN PLATFORM-AGNOSTIC FUNCTION
def extract_intelligent_rules(content: str, source_type: str = "documentation") -> List[Dict]:
    """
    Main function for platform-agnostic rule extraction
    Works with any CI/CD platform, build system, or documentation
    """
    print(f"🧠 Extracting intelligent rules from {source_type}...")
    
    # Extract rules using multiple methods
    rules = extract_platform_agnostic_rules(content, source_type)
    
    # Prioritize and score rules
    prioritized_rules = intelligent_rule_prioritization(rules)
    
    # Remove duplicates and low-quality rules
    filtered_rules = []
    seen_rules = set()
    
    for rule in prioritized_rules:
        rule_key = rule['rule_text'].lower().strip()
        if (rule_key not in seen_rules and 
            rule.get('confidence', 0) > 0.5 and
            len(rule['rule_text']) > 10):
            seen_rules.add(rule_key)
            filtered_rules.append(rule)
    
    print(f"✅ Extracted {len(filtered_rules)} intelligent rules")
    return filtered_rules[:15]  # Limit to top 50 rules


def enhanced_rag_analysis_with_intelligent_rules(pipeline_text: str) -> str:
    """Enhanced RAG analysis using platform-agnostic rule extraction"""
    
    # Get relevant documents from vector store
    docs = vectorstore.similarity_search(pipeline_text, k=15)
    
    all_intelligent_rules = []
    for doc in all_docs:
        rules = extract_lightweight_intelligent_rules(doc.page_content, doc.metadata.get('category', 'general'))
        all_intelligent_rules.extend(rules)
    success_patterns = []
    failure_patterns = []
    
    # Extract intelligent rules from all retrieved documents
    for doc in docs:
        content = doc.page_content
        source_type = doc.metadata.get('category', 'documentation')
        
        # Use platform-agnostic rule extraction
        intelligent_rules = extract_intelligent_rules(content, source_type)
        all_intelligent_rules.extend(intelligent_rules)
        
        # Extract patterns
        if 'success' in content.lower() or 'completed' in content.lower():
            success_patterns.append(f"✅ {content[:80]}...")
        if any(term in content.lower() for term in ['fail', 'error', 'exception']):
            failure_patterns.append(f"❌ {content[:80]}...")
    
    # Group rules by type and priority
    rule_analysis = analyze_rules_against_pipeline(pipeline_text, all_intelligent_rules)
    
    # Generate comprehensive response
    return generate_intelligent_structured_response(
        pipeline_text, all_intelligent_rules, rule_analysis, 
        success_patterns, failure_patterns
    )

class FileTracker:
    """Track MinIO file changes for incremental updates"""
    
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        self.file_metadata = self.load_metadata()
    
    def load_metadata(self) -> dict:
        """Load existing file metadata"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_metadata(self):
        """Save file metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.file_metadata, f, indent=2)
    
    def get_file_hash(self, content: str) -> str:
        """Generate hash for file content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def detect_changes(self, current_files: dict) -> dict:
        """Detect which files have changed"""
        changes = {
            'added': [],
            'modified': [],
            'deleted': [],
            'unchanged': []
        }
        
        # Check for new/modified files
        for file_key, file_info in current_files.items():
            if file_key not in self.file_metadata:
                changes['added'].append(file_key)
            else:
                stored_info = self.file_metadata[file_key]
                if (stored_info.get('last_modified') != file_info['last_modified'] or
                    stored_info.get('size') != file_info['size']):
                    changes['modified'].append(file_key)
                else:
                    changes['unchanged'].append(file_key)
        
        # Check for deleted files
        for file_key in self.file_metadata:
            if file_key not in current_files:
                changes['deleted'].append(file_key)
        
        return changes
    
    def update_file_metadata(self, file_key: str, file_info: dict, content_hash: str):
        """Update metadata for a file"""
        self.file_metadata[file_key] = {
            **file_info,
            'content_hash': content_hash,
            'updated_at': datetime.now().isoformat()
        }
        self.save_metadata()
    
    def remove_file_metadata(self, file_key: str):
        """Remove metadata for deleted file"""
        if file_key in self.file_metadata:
            del self.file_metadata[file_key]  
            self.save_metadata()

# Global file tracker
file_tracker = FileTracker(METADATA_PATH)

def extract_intelligent_rules(content):
    """Intelligently derive rules from ANY documentation content - platform agnostic"""
    rules = []
    lines = content.split('\n')
    
    # Combine lines to get context
    full_text = content.lower()
    
    # Skip Jenkins pipeline logs - they don't contain business rules
    if '[Pipeline]' in content and 'Started by user' in content:
        return []
    
    # Rule 1: Derive disk space requirements (ANY platform)
    if any(term in full_text for term in ['disk space', 'storage', 'gb required', 'gb recommended', 'minimum space']):
        for line in lines:
            if any(term in line.lower() for term in ['disk', 'space', 'gb', 'storage']) and any(num in line for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9']):
                import re
                disk_nums = re.findall(r'(\d+)\s*gb', line.lower())
                if disk_nums:
                    max_disk = max([int(x) for x in disk_nums])
                    rules.append(f"Ensure sufficient disk space (minimum {max_disk}GB required)")
                    break
                # Also catch mentions without specific numbers
                elif any(term in line.lower() for term in ['substantial', 'sufficient', 'adequate']):
                    rules.append(f"Ensure adequate disk space: {line.strip()}")
                    break
    
    # Rule 2: Derive memory requirements (ANY platform)
    if any(term in full_text for term in ['memory', 'ram', 'sufficient ram', 'memory requirement']):
        for line in lines:
            if any(term in line.lower() for term in ['memory', 'ram']) and not '[Pipeline]' in line:
                clean_line = line.strip().replace('**', '').replace('-', '').strip()
                if len(clean_line) > 10:
                    rules.append(f"Ensure adequate memory: {clean_line}")
                    break
    
    # Rule 3: Derive agent/worker requirements (ANY platform)
    if any(term in full_text for term in ['agent', 'worker', 'runner', 'executor', 'labeled', 'node']):
        for line in lines:
            if any(term in line.lower() for term in ['agent', 'worker', 'runner', 'labeled']) and not '[Pipeline]' in line:
                clean_line = line.strip().replace('**', '').replace('-', '').strip()
                if len(clean_line) > 15 and len(clean_line) < 100:
                    rules.append(f"Use appropriate build agent: {clean_line}")
                    break
    
    # Rule 4: Derive timeout requirements (ANY platform)
    if any(term in full_text for term in ['timeout', 'hours', 'execution time', 'time limit', 'expire']):
        for line in lines:
            if any(term in line.lower() for term in ['timeout', 'hour', 'execution', 'expire', 'time']) and not '[Pipeline]' in line:
                clean_line = line.strip().replace('**', '').replace('-', '').strip()
                if len(clean_line) > 15 and len(clean_line) < 150:
                    rules.append(f"Configure appropriate timeout: {clean_line}")
                    break
    
    # Rule 5: Derive environment variable requirements (INTELLIGENT - finds ANY env vars)
    env_vars = []
    for line in lines:
        # Look for patterns like VAR_NAME = value or VAR_NAME: value
        import re
        env_patterns = re.findall(r'([A-Z_]{3,})\s*[=:]\s*([^\n]{1,50})', line)
        for var_name, var_value in env_patterns:
            if not '[Pipeline]' in line and len(var_name) > 2:
                env_vars.append(f"{var_name}: {var_value.strip()}")
    
    if env_vars:
        rules.append(f"Configure required environment variables: {'; '.join(env_vars[:3])}")
    
    # Rule 6: Derive repository/dependency requirements (ANY platform)
    if any(term in full_text for term in ['clone', 'repository', 'git', 'dependency', 'package', 'library']):
        for line in lines:
            if any(term in line.lower() for term in ['clone', 'git', 'repository', 'depend']) and not '[Pipeline]' in line:
                if 'http' in line or 'github' in line.lower() or 'gitlab' in line.lower():
                    clean_line = line.strip()
                    if len(clean_line) < 150:
                        rules.append(f"Ensure repository/dependency access: {clean_line[:100]}...")
                        break
            elif any(term in line.lower() for term in ['package', 'library', 'module']) and not '[Pipeline]' in line:
                clean_line = line.strip()
                if len(clean_line) < 100:
                    rules.append(f"Configure required dependencies: {clean_line}")
                    break
    
    # Rule 7: Derive build tool requirements (INTELLIGENT - detects ANY build tools)
    build_tools = []
    tool_patterns = ['npm', 'yarn', 'maven', 'gradle', 'cmake', 'make', 'bitbake', 'docker', 'kubectl', 'pip', 'composer']
    for tool in tool_patterns:
        if tool in full_text and not '[Pipeline]' in full_text:
            build_tools.append(tool)
    
    if build_tools:
        rules.append(f"Configure build tools: {', '.join(build_tools[:3])}")
    
    # Rule 8: Derive permission/security requirements (ANY platform)
    if any(term in full_text for term in ['permission', 'security', 'access', 'credential', 'auth']):
        for line in lines:
            if any(term in line.lower() for term in ['permission', 'access', 'credential', 'auth']) and not '[Pipeline]' in line:
                clean_line = line.strip()
                if len(clean_line) > 15 and len(clean_line) < 100:
                    rules.append(f"Configure proper permissions: {clean_line}")
                    break
    
    # Rule 9: Derive performance/optimization requirements (ANY platform)
    if any(term in full_text for term in ['performance', 'optimization', 'parallel', 'thread', 'cache', 'speed']):
        for line in lines:
            if any(term in line.lower() for term in ['parallel', 'thread', 'performance', 'cache', 'optim']) and not '[Pipeline]' in line:
                clean_line = line.strip()
                if len(clean_line) > 20 and len(clean_line) < 100:
                    rules.append(f"Configure performance optimization: {clean_line}")
                    break
    
    # Rule 10: Derive cleanup/maintenance requirements (ANY platform)
    if any(term in full_text for term in ['cleanup', 'clean', 'maintenance', 'artifact', 'workspace']):
        for line in lines:
            if any(term in line.lower() for term in ['clean', 'maintenance', 'artifact']) and not '[Pipeline]' in line:
                clean_line = line.strip()
                if len(clean_line) > 15 and len(clean_line) < 100:
                    rules.append(f"Implement cleanup strategy: {clean_line}")
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_rules = []
    for rule in rules:
        if rule not in seen and len(rule) > 10:
            seen.add(rule)
            unique_rules.append(rule)
    
    print(f"📋 Intelligently extracted {len(unique_rules)} platform-agnostic rules")
    for i, rule in enumerate(unique_rules, 1):
        print(f"   RULE {i}: {rule}")
    
    return unique_rules[:10]  # Return up to 10 intelligent rules



def extract_success_failure_patterns(content):
    """Extract what leads to success vs failure from documentation"""
    patterns = {
        'success_indicators': [],
        'failure_indicators': [],
        'best_practices': []
    }
    
    lines = content.split('\n')
    full_text = content.lower()
    
    # Success patterns
    if any(term in full_text for term in ['successful', 'complete', 'performance', 'optimize']):
        for line in lines:
            if any(term in line.lower() for term in ['successful', 'performance', 'speed', 'optimize']):
                patterns['success_indicators'].append(line.strip())
    
    # Failure patterns  
    if any(term in full_text for term in ['fail', 'error', 'issue', 'problem', 'troubleshoot']):
        for line in lines:
            if any(term in line.lower() for term in ['fail', 'error', 'issue', 'problem']):
                patterns['failure_indicators'].append(line.strip())
    
    # Best practices
    if any(term in full_text for term in ['recommend', 'should', 'best', 'consider']):
        for line in lines:
            if any(term in line.lower() for term in ['recommend', 'should', 'consider', 'best']):
                patterns['best_practices'].append(line.strip())
    
    return patterns


def build_confluence_context(confluence_docs):
    """Build intelligent context with mandatory rules formatting."""
    if not confluence_docs:
        return "No build guidelines found in knowledge base"

    all_intelligent_rules = []
    all_patterns = {'success_indicators': [], 'failure_indicators': [], 'best_practices': []}

    for doc in confluence_docs[:10]:
        content = doc.page_content[:1200].strip()
        
        # Extract intelligent rules (platform-agnostic)
        intelligent_rules = extract_intelligent_rules(content)
        if intelligent_rules:
            all_intelligent_rules.extend(intelligent_rules)
        
        # Extract patterns
        patterns = extract_success_failure_patterns(content)
        for key in patterns:
            all_patterns[key].extend(patterns[key])

    # De-dupe rules (preserve order)
    seen = set()
    unique_rules = []
    for r in all_intelligent_rules:
        if r not in seen:
            seen.add(r)
            unique_rules.append(r)

    # Build the string context for debugging/inspection
    context_parts = []
    if unique_rules:
        context_parts.append("=== MANDATORY: USE ONLY THESE RULES ===")
        for i, rule in enumerate(unique_rules[:15], 1):
            context_parts.append(f"MANDATORY_RULE_{i}: {rule}")
        context_parts.append("=== END MANDATORY RULES ===")

    if all_patterns['success_indicators']:
        context_parts.append("\n=== SUCCESS PATTERNS ===")
        for pattern in list(set(all_patterns['success_indicators']))[:5]:
            context_parts.append(f"✅ {pattern}")

    if all_patterns['failure_indicators']:
        context_parts.append("\n=== FAILURE PATTERNS ===")
        for pattern in list(set(all_patterns['failure_indicators']))[:5]:
            context_parts.append(f"❌ {pattern}")

    # RETURN ONLY THE CONTEXT STRING
    return "\n".join(context_parts)


def parse_mandatory_rules_from_context(context_str: str) -> list:
    rules = []
    for line in context_str.splitlines():
        line = line.strip()
        if line.startswith("MANDATORY_RULE_"):
            # everything after the colon is the exact rule text
            parts = line.split(":", 1)
            if len(parts) == 2:
                rules.append(parts[1].strip())
    return rules

def evaluate_rules_against_workflow(rules: list, workflow: str, patterns: dict) -> dict:
    """Evaluate extracted rules against user workflow and return structured analysis"""
    workflow_lower = workflow.lower()
    
    evaluated_rules = []
    satisfied_count = 0
    violated_count = 0
    risk_factors = []
    
    for i, rule in enumerate(rules, 1):
        rule_passed = True
        
        # Check disk space requirements
        if "disk space" in rule.lower() and "gb" in rule.lower():
            import re
            required_gb = re.findall(r'(\d+)\s*gb', rule.lower())
            if required_gb and not any(disk_term in workflow_lower for disk_term in ['disk', 'storage', 'space']):
                rule_passed = False
                risk_factors.append(f"No disk space configuration found for required {required_gb[0]}GB")
        
        # Check memory requirements
        elif "memory" in rule.lower() or "ram" in rule.lower():
            if not any(mem_term in workflow_lower for mem_term in ['memory', 'ram', 'heap', 'xmx']):
                rule_passed = False
                risk_factors.append("Memory configuration not specified")
        
        # Check timeout requirements
        elif "timeout" in rule.lower():
            if not any(time_term in workflow_lower for time_term in ['timeout', 'time', 'hours', 'minutes']):
                rule_passed = False
                risk_factors.append("No timeout configuration found")
        
        # Check environment variables
        elif "environment" in rule.lower() and "variable" in rule.lower():
            if not any(env_term in workflow_lower for env_term in ['env', 'environment', 'export', 'set']):
                rule_passed = False
                risk_factors.append("Environment variables not configured")
        
        # Check build tools
        elif "build tool" in rule.lower():
            tools_in_rule = []
            for tool in ['maven', 'gradle', 'npm', 'yarn', 'docker', 'bitbake', 'make']:
                if tool in rule.lower():
                    tools_in_rule.append(tool)
            
            if tools_in_rule and not any(tool in workflow_lower for tool in tools_in_rule):
                rule_passed = False
                risk_factors.append(f"Required build tools not found: {', '.join(tools_in_rule)}")
        
        # Check repository/dependency access
        elif "repository" in rule.lower() or "dependency" in rule.lower():
            if not any(repo_term in workflow_lower for repo_term in ['git', 'clone', 'checkout', 'pull', 'fetch']):
                rule_passed = False
                risk_factors.append("Repository access not configured")
        
        status = "PASS" if rule_passed else "FAIL"
        evaluated_rules.append(f"{rule} - {status}")
        
        if rule_passed:
            satisfied_count += 1
        else:
            violated_count += 1
    
    # Determine overall prediction
    violation_ratio = violated_count / len(rules) if rules else 0
    if violation_ratio >= 0.5:
        prediction = "FAIL"
        confidence = max(70, int(85 * violation_ratio))
    elif violation_ratio >= 0.3:
        prediction = "HIGH_RISK"
        confidence = max(60, int(75 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(80, int(90 * (1 - violation_ratio)))
    
    return {
        'evaluated_rules': evaluated_rules,
        'satisfied_count': satisfied_count,
        'violated_count': violated_count,
        'risk_factors': risk_factors,
        'prediction': prediction,
        'confidence': confidence
    }


def add_documents_to_vectorstore(documents: list):
    """Add new documents to existing vectorstore"""
    global vectorstore
    
    if not documents:
        return
    
    try:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        if vectorstore is None:
            # Create new vectorstore if none exists
            vectorstore = FAISS.from_documents(documents, embeddings)
            print(f"✅ Created new vectorstore with {len(documents)} documents")
        else:
            # Add to existing vectorstore
            vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to existing vectorstore")
        
        # Save updated vectorstore
        vectorstore.save_local(VECTOR_STORE_PATH)
        
    except Exception as e:
        print(f"❌ Error adding documents to vectorstore: {e}")

def remove_documents_from_vectorstore(file_key: str):
    """Remove documents from vectorstore by source file"""
    global vectorstore
    
    if vectorstore is None:
        return
    
    try:
        # Find document IDs to remove
        ids_to_remove = []
        docs_to_remove = []
        
        for doc_id, doc in vectorstore.docstore._dict.items():
            if doc.metadata.get('source') == file_key:
                ids_to_remove.append(doc_id)
                docs_to_remove.append(doc)
        
        if not ids_to_remove:
            print(f"No documents found for source: {file_key}")
            return
        
        # Remove from docstore
        for doc_id in ids_to_remove:
            del vectorstore.docstore._dict[doc_id]
        
        # Find indices to remove from FAISS index
        indices_to_remove = []
        for index, doc_id in vectorstore.index_to_docstore_id.items():
            if doc_id in ids_to_remove:
                indices_to_remove.append(index)
        
        # Remove from FAISS index
        if indices_to_remove:
            vectorstore.index.remove_ids(np.array(indices_to_remove, dtype=np.int64))
        
        # Update index_to_docstore_id mapping
        new_mapping = {}
        new_index = 0
        for old_index, doc_id in vectorstore.index_to_docstore_id.items():
            if doc_id not in ids_to_remove:
                new_mapping[new_index] = doc_id
                new_index += 1
        
        vectorstore.index_to_docstore_id = new_mapping
        
        # Save updated vectorstore
        vectorstore.save_local(VECTOR_STORE_PATH)
        
        print(f"✅ Removed {len(ids_to_remove)} documents for source: {file_key}")
        
    except Exception as e:
        print(f"❌ Error removing documents from vectorstore: {e}")


# def enforce_structured_format(raw_response: str) -> str:
#     """Force response into structured format"""
    
#     # Check if already in correct format
#     required_fields = [
#         "DETECTED_STACK:", "ESTABLISHED_RULES:", "HISTORICAL_PATTERNS:",
#         "APPLICABLE_RULES:", "SATISFIED_RULES:", "VIOLATED_RULES:",
#         "RISK_FACTORS:", "PREDICTION:", "CONFIDENCE:", "REASONING:"
#     ]
    
#     if all(field in raw_response for field in required_fields):
#         return raw_response
    
#     # Extract key information from narrative response
#     response_lower = raw_response.lower()
    
#     # Detect technologies
#     detected_stack = "yocto-core, jenkins"
#     if "bitbake" in response_lower:
#         detected_stack += ", bitbake"
#     if "docker" in response_lower:
#         detected_stack += ", docker"
    
#     # Determine prediction based on content
#     if any(word in response_lower for word in ["fail", "error", "problem", "issue", "violation"]):
#         prediction = "FAIL"
#         confidence = "70"
#     elif any(word in response_lower for word in ["improvement", "minor", "could be"]):
#         prediction = "HIGH_RISK"
#         confidence = "60"
#     else:
#         prediction = "PASS"
#         confidence = "80"
    
#     # Extract some reasoning
#     reasoning = raw_response[:200].replace('\n', ' ').strip()
#     if len(reasoning) > 150:
#         reasoning = reasoning[:150] + "..."
    
#     # Return properly formatted response
#     return f"""DETECTED_STACK: {detected_stack}
# ESTABLISHED_RULES: Analysis from knowledge base shows build configuration requirements
# HISTORICAL_PATTERNS: Build failures or success due to configuration issues and resource constraints
# APPLICABLE_RULES: Number of Applicable Rules
# SATISFIED_RULES: Number of Satisfied Rules
# VIOLATED_RULES: Number of Violated Rules
# RISK_FACTORS: Configuration inconsistencies, potential resource limitations
# PREDICTION: {prediction}
# CONFIDENCE: {confidence}%
# REASONING: {reasoning}"""

def sanitize_response_content(raw_response: str) -> str:
    """Enhanced sanitization"""
    if not raw_response:
        return "I apologize, but I couldn't generate a proper response."
    
    # Basic sanitization
    sanitized = raw_response.replace('\r', '').replace('\x00', '').strip()
    
    # # Force structured format
    # structured_response = enforce_structured_format(sanitized)
    
    return sanitized


def update_documents_in_vectorstore(file_key: str, new_documents: list):
    """Update documents for a specific file"""
    print(f"Updating documents for: {file_key}")
    
    # Remove old documents
    remove_documents_from_vectorstore(file_key)
    
    # Add new documents
    add_documents_to_vectorstore(new_documents)
    
    print(f"✅ Updated documents for: {file_key}")


def create_or_load_vector_store():
    """Create new vectorstore or load existing one"""
    global vectorstore
    
    try:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Try to load existing vectorstore
        if os.path.exists(VECTOR_STORE_PATH) and not FORCE_REBUILD_ON_STARTUP:
            try:
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("✅ Loaded existing vector store")
                
                # Perform incremental update check
                perform_incremental_update()
                return
                
            except Exception as e:
                print(f"❌ Could not load existing vectorstore: {e}")
                print("Creating fresh vectorstore...")
        
        # Create fresh vectorstore if loading failed or forced rebuild
        create_vectorstore()
        
    except Exception as e:
        print(f"❌ Error in create_or_load_vector_store: {e}")
        raise

def perform_incremental_update():
    """Check for file changes and perform incremental updates"""
    global vectorstore
    
    print("🔍 Checking for file changes...")
    
    try:
        # Get current MinIO files
        current_files = get_minio_file_snapshot()
        
        # Detect changes
        changes = file_tracker.detect_changes(current_files)
        
        print(f"📊 Change detection results:")
        print(f"  Added: {len(changes['added'])} files")
        print(f"  Modified: {len(changes['modified'])} files")
        print(f"  Deleted: {len(changes['deleted'])} files")
        print(f"  Unchanged: {len(changes['unchanged'])} files")
        
        if not any([changes['added'], changes['modified'], changes['deleted']]):
            print("✅ No changes detected - vectorstore is up to date")
            return
        
        # Process changes
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            verify=False
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=25
        )
        
        # Handle deleted files
        for file_key in changes['deleted']:
            print(f" Removing documents for deleted file: {file_key}")
            remove_documents_from_vectorstore(file_key)
            file_tracker.remove_file_metadata(file_key)
        
        # Handle added files
        for file_key in changes['added']:
            print(f" Processing new file: {file_key}")
            process_single_file(s3, file_key, current_files[file_key], text_splitter, 'add')
        
        # Handle modified files
        for file_key in changes['modified']:
            print(f" Processing modified file: {file_key}")
            process_single_file(s3, file_key, current_files[file_key], text_splitter, 'update')
        
        print("✅ Incremental update completed")
        
    except Exception as e:
        print(f"❌ Error in incremental update: {e}")

def process_single_file(s3_client, file_key: str, file_info: dict, text_splitter, operation: str):
    """Process a single file for add/update operations"""
    try:
        # Download file content
        file_obj = s3_client.get_object(Bucket=MINIO_BUCKET, Key=file_key)
        file_content = file_obj["Body"].read().decode("utf-8", errors="ignore")
        
        if not file_content.strip():
            print(f"⚠️ Empty file skipped: {file_key}")
            return
        
        # Split into chunks
        chunks = text_splitter.split_text(file_content)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_key,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "file_size": file_info["size"],
                        "last_modified": file_info["last_modified"]
                    }
                )
                documents.append(doc)
        
        # Add or update documents
        if operation == 'add':
            add_documents_to_vectorstore(documents)
        elif operation == 'update':
            update_documents_in_vectorstore(file_key, documents)
        
        # Update file tracker
        content_hash = file_tracker.get_file_hash(file_content)
        file_tracker.update_file_metadata(file_key, file_info, content_hash)
        
        print(f"✅ Processed {len(documents)} chunks from {file_key}")
        
    except Exception as e:
        print(f"❌ Error processing file {file_key}: {e}")




class ChatRequest(BaseModel):
    messages: list
    model: str = "codellama:7b"
    temperature: float = 0.4
    max_tokens: int = 2000


class CompletionRequest(BaseModel):
    prompt: str
    model: str = "codellama:7b"
    temperature: float = 0.4
    max_tokens: int = 2000


class CompletionResponse(BaseModel):
    choices: list
    model: str
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# ===== DATABASE CONFIGURATION =====
DB_CONFIG = {
    'host': 'localhost',
    'database': 'rag_feedback',
    'user': 'rag_user',
    'password': 'eic@123456', 
    'port': 5432
}

# Create connection pool
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        **DB_CONFIG
    )
    print("✅ PostgreSQL connection pool created")
except Exception as e:
    print(f"⚠️  PostgreSQL connection failed: {e}")
    connection_pool = None

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    if connection_pool is None:
        raise Exception("Database connection pool not initialized")
    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        connection_pool.putconn(conn)

def get_db_cursor(conn):
    """Get cursor that returns results as dictionaries"""
    return conn.cursor(cursor_factory=RealDictCursor)

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_build_result: str  # SUCCESS/FAILURE
    corrected_confidence: Optional[int] = None
    missed_issues: List[str] = []
    false_positives: List[str] = []
    user_comments: Optional[str] = None
    suggested_rules: Optional[List[dict]] = None
    feedback_type: str = "manual" 


# ========= Build Lock Management =========
def acquire_build_lock():
    """Acquire global build lock across all workers"""
    try:
        with GLOBAL_BUILD_LOCK:
            if os.path.exists(BUILD_LOCK_FILE):
                lock_age = time.time() - os.path.getmtime(BUILD_LOCK_FILE)
                if lock_age > 300:  # 5 minutes
                    print("Removing stale build lock")
                    os.remove(BUILD_LOCK_FILE)
                else:
                    return False
            
            with open(BUILD_LOCK_FILE, 'w') as f:
                f.write(str(os.getpid()))
            return True
    except Exception as e:
        print(f"❌ Error acquiring build lock: {e}")
        return False


def release_build_lock():
    """Release global build lock"""
    try:
        if os.path.exists(BUILD_LOCK_FILE):
            os.remove(BUILD_LOCK_FILE)
    except Exception as e:
        print(f"❌ Error releasing build lock: {e}")


# ========= MinIO File Monitoring Functions =========
def get_minio_file_snapshot():
    """Get current snapshot of MinIO files with their metadata"""
    try:
        print("Connecting to MinIO to get file snapshot...")
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            verify=False
        )
        
        response = s3.list_objects_v2(Bucket=MINIO_BUCKET)
        objects = response.get("Contents", [])
        
        print(f"MinIO Connection Success - Found {len(objects)} files in bucket '{MINIO_BUCKET}':")
        
        snapshot = {}
        for obj in objects:
            snapshot[obj["Key"]] = {
                "last_modified": obj["LastModified"].isoformat(),
                "size": obj["Size"],
                "etag": obj.get("ETag", "").replace('"', '')
            }
            print(f"  - {obj['Key']} (Size: {obj['Size']} bytes, Modified: {obj['LastModified']})")
        
        return snapshot
    except Exception as e:
        print(f"❌ Error connecting to MinIO or getting snapshot: {e}")
        return {}


def verify_minio_connection():
    """Verify MinIO connection and show current state"""
    try:
        snapshot = get_minio_file_snapshot()
        print(f"MinIO verification: {len(snapshot)} files found")
        return len(snapshot) > 0
    except Exception as e:
        print(f"❌ MinIO verification failed: {e}")
        return False


class MinIOFileWatcher(FileSystemEventHandler):
    """File system event handler for MinIO data directory changes"""
    
    def __init__(self):
        super().__init__()
        self.loop = None
    
    def set_loop(self, loop):
        self.loop = loop
    
    def on_modified(self, event):
        if not event.is_directory and self._is_user_content_file(event.src_path):
            print(f"User file modified: {event.src_path}")
            self._schedule_refresh()
    
    def on_created(self, event):
        if not event.is_directory and self._is_user_content_file(event.src_path):
            print(f"User file created: {event.src_path}")
            self._schedule_refresh()
    
    def on_deleted(self, event):
        if not event.is_directory and self._is_user_content_file(event.src_path):
            print(f"User file deleted: {event.src_path}")
            self._schedule_refresh()
    
    def _is_user_content_file(self, file_path):
        """Check if the file is actual user content, not MinIO system files"""
        try:
            normalized_path = file_path.replace('\\', '/')
            
            # Skip MinIO system files completely
            skip_patterns = [
                '.minio.sys/',
                'xl.meta',
                '.usage-cache',
                '/tmp/',
                '.tmp',
                'part.minio',
                '.backup',
                '.bkp',
                '/.trash/',
                '/multipart/'
            ]
            
            for pattern in skip_patterns:
                if pattern in normalized_path:
                    return False
            
            # Only consider files in our bucket
            if f"/{MINIO_BUCKET}/" in normalized_path:
                filename = os.path.basename(file_path)
                
                if filename.startswith('.') or filename.endswith('.tmp'):
                    return False
                
                if not filename.strip():
                    return False
                
                print(f"User content file detected: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error checking file: {e}")
            return False
    
    def _schedule_refresh(self):
        global last_change_time, refresh_task
        
        last_change_time = time.time()
        print(f"User content change detected, scheduling refresh...")
        
        if refresh_task and not refresh_task.done():
            refresh_task.cancel()
        
        if self.loop:
            refresh_task = asyncio.run_coroutine_threadsafe(
                self._debounced_refresh(), self.loop
            )
    
    async def _debounced_refresh(self):
        try:
            print(f"Waiting {DEBOUNCE_SECONDS} seconds for more changes...")
            await asyncio.sleep(DEBOUNCE_SECONDS)
            
            if time.time() - last_change_time >= DEBOUNCE_SECONDS:
                print("Debounce completed, refreshing vector store...")
                await refresh_vector_store_if_needed(force_refresh=True)
            
        except asyncio.CancelledError:
            print("Refresh task cancelled due to new changes")
        except Exception as e:
            print(f"❌ Error in debounced refresh: {e}")


def setup_file_watcher(loop):
    """Setup file system watcher for MinIO data directory"""
    global file_observer
    
    if not WATCH_ENABLED:
        print("File watching is disabled")
        return None
    
    try:
        watch_path = os.path.abspath(os.path.expanduser(MINIO_DATA_PATH))
        
        if not os.path.exists(watch_path):
            print(f"MinIO data path does not exist: {watch_path}")
            return None
        
        event_handler = MinIOFileWatcher()
        event_handler.set_loop(loop)
        
        file_observer = Observer()
        file_observer.schedule(event_handler, watch_path, recursive=True)
        file_observer.start()
        
        print(f"File watcher started for: {watch_path}")
        return file_observer
        
    except Exception as e:
        print(f"❌ Could not setup file watcher: {e}")
        return None


def stop_file_watcher():
    """Stop the file watcher"""
    global file_observer, refresh_task
    
    if file_observer:
        print("Stopping file watcher...")
        file_observer.stop()
        file_observer.join()
        file_observer = None
    
    if refresh_task and not refresh_task.done():
        refresh_task.cancel()


async def refresh_vector_store_if_needed(force_refresh=False):
    """Refresh vector store when file changes are detected"""
    global vectorstore, retriever
    
    if refresh_lock.locked() and not force_refresh:
        print("Refresh already in progress, skipping...")
        return False
    
    async with refresh_lock:
        try:
            print("Starting vector store refresh...")
            
            if not acquire_build_lock():
                print("Another process is building, waiting...")
                for i in range(30):
                    await asyncio.sleep(1)
                    if not os.path.exists(BUILD_LOCK_FILE):
                        break
                
                await run_in_threadpool(load_existing_vectorstore)
                return True
            
            try:
                refresh_start_time = time.time()
                
                # Always force delete old vector store for complete rebuild
                if os.path.exists(VECTOR_STORE_PATH):
                    print("Removing old vector store for fresh rebuild...")
                    shutil.rmtree(VECTOR_STORE_PATH)
                
                await run_in_threadpool(create_vectorstore)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
                
                refresh_time = time.time() - refresh_start_time
                print(f"✅ Vector store completely rebuilt in {refresh_time:.2f}s")
                return True
                
            finally:
                release_build_lock()
            
        except Exception as e:
            print(f"❌ Error refreshing vector store: {e}")
            release_build_lock()
            return False


def load_existing_vectorstore():
    """Load existing vectorstore without rebuilding"""
    global vectorstore
    
    try:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        if os.path.exists(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing vector store")
        else:
            print("No existing vector store found")
            
    except Exception as e:
        print(f"❌ Error loading existing vectorstore: {e}")


# ========= Graceful Shutdown Management =========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle WITHOUT deleting vectorstore"""
    print(" Starting RAG API Service with persistent vectorstore...")
    
    # Verify MinIO connection
    if not verify_minio_connection():
        print("❌ Cannot connect to MinIO - exiting")
        raise Exception("MinIO connection failed")
    
    # Initialize system with persistent vectorstore
    await initialize_rag_system_async()
    
    # Setup file watcher for incremental updates
    loop = asyncio.get_event_loop()
    setup_file_watcher(loop)
    
    print("✅ RAG API Service ready with persistent vectorstore!")
    
    yield
    
    print("Shutting down RAG API Service...")
    shutdown_event.set()
    stop_file_watcher()
    release_build_lock()
    
    print("✅ RAG API Service stopped gracefully (vectorstore preserved)")


async def refresh_vector_store_if_needed(force_refresh=False):
    """Perform incremental refresh instead of full rebuild"""
    global vectorstore, retriever
    
    if refresh_lock.locked() and not force_refresh:
        print("Refresh already in progress, skipping...")
        return False
    
    async with refresh_lock:
        try:
            print("🔄 Starting incremental vector store refresh...")
            
            if not acquire_build_lock():
                print("Another process is updating, waiting...")
                return False
            
            try:
                refresh_start_time = time.time()
                
                # Perform incremental update instead of full rebuild
                await run_in_threadpool(perform_incremental_update)
                
                # Update retriever
                if vectorstore:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
                
                refresh_time = time.time() - refresh_start_time
                print(f"✅ Incremental update completed in {refresh_time:.2f}s")
                return True
                
            finally:
                release_build_lock()
            
        except Exception as e:
            print(f"❌ Error in incremental refresh: {e}")
            release_build_lock()
            return False

# Initialize FastAPI with lifespan management
app = FastAPI(title="RAG API Service", version="1.0.0", lifespan=lifespan)


# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000", 
    "http://localhost:8000",
    "https://continue.dev",
    "vscode-webview://*",
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= User Management Functions =========
def get_user_id(request: Request, x_user_id: Optional[str] = None) -> str:
    if x_user_id:
        return x_user_id
    
    user_agent = request.headers.get("user-agent", "")
    if "vscode" in user_agent.lower() or "continue" in user_agent.lower():
        client_ip = request.client.host if request.client else "unknown"
        session_id = f"vscode_{client_ip}_{hash(user_agent) % 10000:04d}"
        return session_id
    
    client_ip = request.client.host if request.client else "unknown"
    return f"user_{client_ip}"


async def get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]


async def check_user_request_limit(user_id: str) -> bool:
    lock = await get_user_lock(user_id)
    return lock.locked()


# ========= RAG System Functions =========
async def initialize_rag_system_async():
    """Initialize the RAG system components asynchronously"""
    await run_in_threadpool(initialize_rag_system)


def initialize_rag_system():
    """Initialize the RAG system with persistent vectorstore"""
    global vectorstore, retriever, llm
    
    try:
        print("🚀 Initializing RAG system with persistent vectorstore...")
        
        # Create or load vectorstore (no more deletion!)
        create_or_load_vector_store()
        
        # Initialize retriever and LLM
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
        
        print("✅ RAG system initialized with persistent data")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        raise

def extract_lightweight_intelligent_rules(content: str, source_type: str = "documentation") -> List[Dict]:
    """
    Lightweight platform-agnostic rule extraction using regex patterns
    No external dependencies required - works with any platform/documentation
    """
    rules = []
    
    print(f"🧠 Extracting intelligent rules from {source_type}...")
    
    # 1. REQUIREMENT EXTRACTION (Modal Verbs)
    rules.extend(extract_requirement_rules(content))
    
    # 2. CONFIGURATION EXTRACTION  
    rules.extend(extract_configuration_rules(content))
    
    # 3. CONSTRAINT EXTRACTION
    rules.extend(extract_constraint_rules_lightweight(content))
    
    # 4. PROCESS STEP EXTRACTION
    rules.extend(extract_process_rules(content))
    
    # 5. DEPENDENCY EXTRACTION
    rules.extend(extract_dependency_rules_lightweight(content))
    
    # Remove duplicates and sort by confidence
    unique_rules = {}
    for rule in rules:
        key = rule['rule_text'].lower().strip()
        if key not in unique_rules:
            unique_rules[key] = rule
        elif rule.get('confidence', 0) > unique_rules[key].get('confidence', 0):
            unique_rules[key] = rule
    
    sorted_rules = sorted(unique_rules.values(), key=lambda x: x.get('confidence', 0.5), reverse=True)
    
    print(f"✅ Extracted {len(sorted_rules)} intelligent rules")
    return sorted_rules[:30]  # Limit to top 30

def extract_requirement_rules(content: str) -> List[Dict]:
    """Extract requirement rules using modal verbs and imperative language"""
    rules = []
    lines = content.split('\n')
    
    # Requirement patterns with confidence scores
    requirement_patterns = [
        (r'\bmust\b.*', 'MANDATORY', 0.95),
        (r'\bshall\b.*', 'MANDATORY', 0.90),
        (r'\brequired?\b.*', 'MANDATORY', 0.85),
        (r'\bmandatory\b.*', 'MANDATORY', 0.90),
        (r'\bshould\b.*', 'RECOMMENDED', 0.75),
        (r'\brecommended?\b.*', 'RECOMMENDED', 0.70),
        (r'\bensure\b.*', 'VERIFICATION', 0.80),
        (r'\bverify\b.*', 'VERIFICATION', 0.80),
        (r'\bconfigure\b.*', 'CONFIGURATION', 0.75),
        (r'\bsetup?\b.*', 'SETUP', 0.70)
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 15 or len(line) > 200:
            continue
            
        line_lower = line.lower()
        
        for pattern, rule_type, confidence in requirement_patterns:
            if re.search(pattern, line_lower):
                rules.append({
                    'rule_text': line,
                    'rule_type': rule_type,
                    'source': 'REQUIREMENT_ANALYSIS',
                    'confidence': confidence
                })
                break
    
    return rules

def extract_configuration_rules(content: str) -> List[Dict]:
    """Extract configuration and environment rules"""
    rules = []
    
    # Configuration patterns
    config_patterns = [
        (r'[A-Z_]+\s*=\s*.+', 'ENVIRONMENT_VAR', 0.85),  # ENV_VAR = value
        (r'export\s+[A-Z_]+', 'ENVIRONMENT_EXPORT', 0.80),
        (r'\.conf\b', 'CONFIG_FILE', 0.75),
        (r'\.properties\b', 'CONFIG_FILE', 0.75),
        (r'\.yaml\b|\.yml\b', 'CONFIG_FILE', 0.75),
        (r'\.json\b', 'CONFIG_FILE', 0.70),
        (r'port\s*:\s*\d+', 'PORT_CONFIG', 0.80),
        (r'timeout\s*[:=]\s*\d+', 'TIMEOUT_CONFIG', 0.80)
    ]
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) < 5 or len(line) > 150:
            continue
            
        for pattern, rule_type, confidence in config_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                rules.append({
                    'rule_text': line,
                    'rule_type': rule_type,
                    'source': 'CONFIG_ANALYSIS',
                    'confidence': confidence
                })
                break
    
    return rules

def extract_constraint_rules_lightweight(content: str) -> List[Dict]:
    """Extract system constraints and limits"""
    rules = []
    
    # Resource constraint patterns
    constraint_patterns = [
        (r'(\d+)\s*(GB|MB|TB|gb|mb|tb)', 'STORAGE_REQUIREMENT', 0.90),
        (r'(\d+)\s*(hours?|hrs?|minutes?|mins?)', 'TIME_LIMIT', 0.85),
        (r'(\d+)\s*(cores?|threads?|CPUs?)', 'COMPUTE_REQUIREMENT', 0.85),
        (r'minimum.*?(\d+)', 'MINIMUM_REQUIREMENT', 0.80),
        (r'maximum.*?(\d+)', 'MAXIMUM_LIMIT', 0.80),
        (r'at\s+least.*?(\d+)', 'MINIMUM_REQUIREMENT', 0.75),
        (r'memory.*?(\d+)', 'MEMORY_REQUIREMENT', 0.80)
    ]
    
    for pattern, constraint_type, confidence in constraint_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            context_start = max(0, match.start() - 50)
            context_end = min(len(content), match.end() + 50)
            context = content[context_start:context_end].strip()
            
            rules.append({
                'rule_text': context,
                'rule_type': constraint_type,
                'source': 'CONSTRAINT_ANALYSIS',
                'confidence': confidence,
                'value': match.group(1) if match.groups() else match.group(0)
            })
    
    return rules

def extract_process_rules(content: str) -> List[Dict]:
    """Extract process and procedural rules"""
    rules = []
    lines = content.split('\n')
    
    # Process step patterns
    step_patterns = [
        (r'^\d+\.\s+(.+)$', 'NUMBERED_STEP', 0.85),
        (r'^step\s+\d+[:\-]\s*(.+)$', 'EXPLICIT_STEP', 0.90),
        (r'^\w+[\)\]]\s+(.+)$', 'LETTERED_STEP', 0.80),
        (r'^[•\-\*]\s+(.+)$', 'BULLET_STEP', 0.70),
        (r'first[,\s]', 'SEQUENCE_STEP', 0.75),
        (r'then[,\s]', 'SEQUENCE_STEP', 0.75),
        (r'next[,\s]', 'SEQUENCE_STEP', 0.75),
        (r'finally[,\s]', 'SEQUENCE_STEP', 0.75)
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
            
        for pattern, rule_type, confidence in step_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                rules.append({
                    'rule_text': line,
                    'rule_type': rule_type,
                    'source': 'PROCESS_ANALYSIS',
                    'confidence': confidence
                })
                break
    
    return rules

def extract_dependency_rules_lightweight(content: str) -> List[Dict]:
    """Extract dependency and prerequisite rules"""
    rules = []
    
    # Dependency patterns
    dependency_patterns = [
        (r'depends\s+on\s+(.+)', 'DEPENDENCY', 0.85),
        (r'requires?\s+(.+)', 'REQUIREMENT', 0.80),
        (r'needs?\s+(.+)', 'REQUIREMENT', 0.75),
        (r'prerequisite[:\s]+(.+)', 'PREREQUISITE', 0.85),
        (r'before\s+(.+)', 'SEQUENCE_DEPENDENCY', 0.70),
        (r'after\s+(.+)', 'SEQUENCE_DEPENDENCY', 0.70),
        (r'must\s+have\s+(.+)', 'REQUIREMENT', 0.80),
        (r'install\s+(.+)', 'INSTALLATION', 0.75),
        (r'setup\s+(.+)', 'SETUP', 0.70)
    ]
    
    lines = content.split('\n')
    for line in lines:
        if len(line.strip()) < 10:
            continue
            
        line_lower = line.lower()
        for pattern, rule_type, confidence in dependency_patterns:
            match = re.search(pattern, line_lower)
            if match:
                dependency = match.group(1).strip()
                if len(dependency) > 3 and len(dependency) < 100:
                    rules.append({
                        'rule_text': f'{rule_type}: {dependency}',
                        'rule_type': rule_type,
                        'source': 'DEPENDENCY_ANALYSIS',
                        'confidence': confidence,
                        'dependency': dependency
                    })
                break
    
    return rules

def create_vectorstore():
    """Create vector store from MinIO documents with proper categorization and chunk tracking"""
    global vectorstore
    print("=" * 50)
    print("CREATING COMPLETELY FRESH VECTOR STORE WITH CATEGORIZATION")
    print("=" * 50)
    
    s3 = boto3.client('s3', 
                     endpoint_url=MINIO_ENDPOINT,
                     aws_access_key_id=MINIO_ACCESS_KEY,
                     aws_secret_access_key=MINIO_SECRET_KEY,
                     verify=False)
    
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET)
    objects = response.get('Contents', [])
    print(f"FOUND {len(objects)} FILES IN MINIO")
    
    if not objects:
        raise Exception(f"NO FILES IN BUCKET {MINIO_BUCKET}")
    
    documents = []
    
    # Different text splitters for different content types
    confluence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # Larger chunks for documentation
        chunk_overlap=50
    )
    
    log_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    # Smaller chunks for logs
        chunk_overlap=25
    )
    
    # Categorization counters
    confluence_docs = 0
    confluence_chunks = 0
    log_docs = 0
    log_chunks = 0
    
    for obj in objects:
        file_key = obj['Key']
        print(f"PROCESSING: {file_key}")
        
        # Get file content
        file_obj = s3.get_object(Bucket=MINIO_BUCKET, Key=file_key)
        file_content = file_obj['Body'].read().decode('utf-8', errors='ignore')
            # DEBUG: Check condition
        print(f"   Is static_knowledge: {'static_knowledge' in file_key.lower()}")
        print(f"   Ends with .md: {file_key.endswith('.md')}")
        
        # Load page_metadata.json for Confluence URLs (if exists)
        page_metadata_extra = {}
        if 'static_knowledge' in file_key.lower() and file_key.endswith('.md'):
            # Try to find corresponding page_metadata.json in same folder
            folder_path = '/'.join(file_key.split('/')[:-1])  # Get folder path
            metadata_key = f"{folder_path}/page_metadata.json"
            
            try:
                metadata_obj = s3.get_object(Bucket=MINIO_BUCKET, Key=metadata_key)
                metadata_content = metadata_obj['Body'].read().decode('utf-8')
                page_metadata_extra = json.loads(metadata_content)
                print(f"   Loaded metadata with URL: {page_metadata_extra.get('confluence_url', 'N/A')[:80]}...")
            except Exception as e:
                print(f"   No metadata file found at {metadata_key}")
        
        # CATEGORIZE FILES BASED ON PATH AND CONTENT
        if 'dynamic_knowledge' in file_key.lower():
            # This is a LOG file
            category = 'log'
            file_type = 'jenkins-log'
            chunks = log_splitter.split_text(file_content)
            print(f"   LOG FILE: Created {len(chunks)} chunks")
            log_docs += 1
            log_chunks += len(chunks)
            
        elif 'static_knowledge' in file_key.lower() or file_key.endswith('.md'):
            # This is CONFLUENCE documentation
            category = 'documentation'
            file_type = 'confluence-doc'
            chunks = confluence_splitter.split_text(file_content)
            print(f"   CONFLUENCE DOC: Created {len(chunks)} chunks")
            confluence_docs += 1
            confluence_chunks += len(chunks)
            
        elif any(ext in file_key.lower() for ext in ['.bb', '.bbappend', '.conf', '.inc', '.bbclass']):
            if file_key.endswith(('.bb', '.bbappend')):
                category = "yocto-recipe"
                file_type = "bitbake-recipe"
            elif 'layer.conf' in file_key or 'bblayers.conf' in file_key or 'local.conf' in file_key:
                category = "yocto-config"
                file_type = "yocto-layer-config"
            else:
                category = "yocto-config"
                file_type = "yocto-general-config"
            
            chunks = confluence_splitter.split_text(file_content)
            print(f"   YOCTO FILE ({file_type}): Created {len(chunks)} chunks")
            confluence_docs += 1
            confluence_chunks += len(chunks)
            
        else:
            # Default categorization
            category = 'general'
            file_type = 'unknown'
            chunks = confluence_splitter.split_text(file_content)
            print(f"   GENERAL FILE: Created {len(chunks)} chunks")
            confluence_docs += 1
            confluence_chunks += len(chunks)
        
        # Create documents with enhanced metadata including Confluence URLs
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': file_key,
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'category': category,
                        'file_type': file_type,
                        'file_size': len(file_content),
                        # Add Confluence URL and page metadata
                        'confluence_url': page_metadata_extra.get('confluence_url', ''),
                        'page_id': page_metadata_extra.get('page_id', ''),
                        'page_title': page_metadata_extra.get('page_title', ''),
                        'original_title': page_metadata_extra.get('original_title', ''),
                        'space_key': page_metadata_extra.get('space_key', '')
                    }
                )
                documents.append(doc)
    
    print(f"\nCATEGORIZATION SUMMARY:")
    print(f"   Confluence Docs: {confluence_docs} files -> {confluence_chunks} chunks")
    print(f"   Log Files: {log_docs} files -> {log_chunks} chunks")
    print(f"   Total Documents: {len(documents)} chunks")
    
    # Create embeddings and vector store
    print(f"\nCREATING EMBEDDINGS FOR {len(documents)} CHUNKS...")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    
    # Save build metadata with categorization info
    build_metadata = {
        'build_datetime': datetime.now().isoformat(),
        'total_chunks': len(documents),
        'confluence_docs': confluence_docs,
        'confluence_chunks': confluence_chunks,
        'log_docs': log_docs,
        'log_chunks': log_chunks,
        'files_processed': len(objects)
    }
    
    metadata_file = os.path.join(VECTOR_STORE_PATH, "build_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(build_metadata, f, indent=2)
    
    print(f"\nFRESH VECTOR STORE CREATED:")
    print(f"   {confluence_chunks} Confluence chunks from {confluence_docs} documents")
    print(f"   {log_chunks} Log chunks from {log_docs} files")
    print(f"   {len(documents)} total chunks indexed")
    print("=" * 50)


# def sanitize_response_content(raw_response: str) -> str:
#     """Sanitize but do NOT alter a valid structured response."""
#     if not raw_response:
#         return "I apologize, but I couldn't generate a proper response."
    
#     cleaned = raw_response.replace('\r', '').replace('\x00', '').strip()

#     # If already structured, leave untouched
#     if all(tag in cleaned for tag in [
#         "DETECTED_STACK:", "ESTABLISHED_RULES:", "HISTORICAL_PATTERNS:",
#         "APPLICABLE_RULES:", "SATISFIED_RULES:", "VIOLATED_RULES:",
#         "RISK_FACTORS:", "PREDICTION:", "CONFIDENCE:", "REASONING:"
#     ]):
#         return cleaned
    
#     # Otherwise return as-is (no fallback narrative rewriting)
#     return cleaned


# def sanitize_response_content(raw_response: str) -> str:
#     if not raw_response:
#         return "I apologize, but I couldn't generate a proper response."
    
#     sanitized = raw_response.replace('\r', '').replace('\x00', '').strip()
#     return sanitized
def extract_success_patterns_from_logs(content: str) -> List[str]:
    """Enhanced success pattern extraction"""
    success_patterns = []
    lines = content.split('\n')
    
    success_indicators = [
        'success', 'completed successfully', 'build succeeded', 'finished successfully',
        'build #', 'duration:', 'yocto image build succeeded', 'pipeline succeeded',
        'archived', 'deployed', 'published', 'passed', 'ok', 'done'
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
            
        line_lower = line.lower()
        for indicator in success_indicators:
            if indicator in line_lower:
                success_patterns.append(f"✅ {line}")
                break
    
    return success_patterns[:10]


def get_historical_patterns_from_logs():
    """Get historical patterns by searching specifically for log documents"""
    print("Searching for historical patterns in LOG documents...")
    
    success_patterns = []
    failure_patterns = []
    
    # Get ALL documents from vector store
    try:
        # Search specifically for log content
        log_search_queries = [
            "jenkins build log",
            "dynamic_knowledge jenkins_logs", 
            "yocto-build-14 yocto-build-18",
            "build_numbers summary",
            "console output",
            "pipeline finished"
        ]
        
        log_documents = []
        for query in log_search_queries:
            docs = vectorstore.similarity_search(query, k=5)
            for doc in docs:
                # Filter for LOG documents specifically
                if doc.metadata.get('category') == 'log' or 'dynamic_knowledge' in doc.metadata.get('source', ''):
                    log_documents.append(doc)
        
        # Remove duplicates
        unique_log_docs = []
        seen_sources = set()
        for doc in log_documents:
            source = doc.metadata.get('source', '')
            if source not in seen_sources:
                seen_sources.add(source)
                unique_log_docs.append(doc)
        
        print(f"Processing {len(unique_log_docs)} unique LOG documents")
        
        # Extract patterns from log documents
        for doc in unique_log_docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            
            print(f"  Analyzing log: {source}")
            
            doc_success = extract_success_patterns_from_logs(content)
            doc_failure = extract_failure_patterns_from_logs(content)
            
            success_patterns.extend(doc_success)
            failure_patterns.extend(doc_failure)
            
            if doc_success or doc_failure:
                print(f"    Extracted {len(doc_success)} success + {len(doc_failure)} failure patterns")
        
        # Remove duplicates and limit
        success_patterns = list(set(success_patterns))[:5]
        failure_patterns = list(set(failure_patterns))[:5]
        
        print(f"✅ Final result: {len(success_patterns)} success patterns, {len(failure_patterns)} failure patterns")
        
    except Exception as e:
        print(f"Error extracting log patterns: {e}")
        success_patterns = ["Log pattern extraction failed"]
        failure_patterns = ["Log pattern extraction failed"]
    
    return success_patterns, failure_patterns


def extract_failure_patterns_from_logs(content: str) -> List[str]:
    """Enhanced failure pattern extraction"""  
    failure_patterns = []
    lines = content.split('\n')
    
    failure_indicators = [
        'error', 'failed', 'exception', 'timeout', 'aborted', 'cancelled',
        'build failed', 'pipeline failed', 'yocto build failed', 'bitbake failed',
        'permission denied', 'no space', 'connection refused', 'not found'
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
            
        line_lower = line.lower()
        for indicator in failure_indicators:
            if indicator in line_lower:
                failure_patterns.append(f"❌ {line}")
                break
    
    return failure_patterns[:10]


def generate_intelligent_suggestions_from_violations(rule_analysis: Dict, query: str) -> List[str]:
    """
    Generate intelligent suggestions by analyzing ACTUAL violated rules using AI
    WITH SOURCE CITATIONS for each suggestion
    """
    if not rule_analysis or not rule_analysis.get('critical_violations') and not rule_analysis.get('warnings'):
        return []
    
    # Collect all violated rules
    violated_rules = []
    violated_rules.extend(rule_analysis.get('critical_violations', [])[:5])  # Top 5 critical
    violated_rules.extend(rule_analysis.get('warnings', [])[:3])            # Top 3 warnings
    
    if not violated_rules:
        return []
    
    print(f"🔍 Generating AI suggestions for {len(violated_rules)} violated rules...")
    
    #  NEW: Build context WITH CITATIONS from violated rules
    violation_context_with_citations = []
    for rule in violated_rules:
        if isinstance(rule, dict):
            rule_text = rule.get('rule_text', str(rule))
            citation = rule.get('citation', '')
            
            violation_context_with_citations.append(f"• {rule_text}")
            if citation:
                violation_context_with_citations.append(f"  Source: {citation}")
        else:
            violation_context_with_citations.append(f"• {rule}")
    
    violation_context = '\n'.join(violation_context_with_citations)
    
    #  ENHANCED: AI prompt that includes source references
    suggestion_prompt = f"""Based on these SPECIFIC violated rules from pipeline analysis:

VIOLATED RULES (WITH SOURCES):
{violation_context}

PIPELINE CONTEXT:
{query[:800]}...

Generate 3-5 SPECIFIC, ACTIONABLE suggestions to fix these violations and improve pipeline success rate.

IMPORTANT: For each suggestion, reference the source documentation if provided above.

Format each suggestion as:
• [Detailed fix description with commands/configuration]
  Source: [Reference the source if available]
  Impact: (+X% confidence improvement)

Focus on:
1. Configuration fixes (exact files, settings, parameters)
2. Missing dependencies or tools with installation commands
3. Resource allocation issues with specific values
4. Process improvements with implementation steps

Make suggestions concrete, implementable, and reference the source documentation."""

    try:
        # Use your existing LLM to generate suggestions
        ai_response = llm.invoke(suggestion_prompt)
        
        #  ENHANCED: Parse AI response and preserve source citations
        suggestions = []
        current_suggestion = ""
        
        for line in ai_response.split('\n'):
            line = line.strip()
            
            # Start of new suggestion
            if line.startswith('•'):
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())
                current_suggestion = line[1:].strip()  # Remove bullet
            
            # Continuation of current suggestion (Source: or Impact:)
            elif line.startswith(('Source:', 'Impact:')) and current_suggestion:
                current_suggestion += f"\n   {line}"
            
            # Multi-line suggestion content
            elif current_suggestion and line and not line.startswith('•'):
                current_suggestion += f" {line}"
        
        # Don't forget the last suggestion
        if current_suggestion:
            suggestions.append(current_suggestion.strip())
        
        print(f"✅ Generated {len(suggestions)} AI-driven suggestions with source citations")
        return suggestions[:5]  # Limit to 5 suggestions
        
    except Exception as e:
        print(f"⚠️ AI suggestion generation failed: {e}")
        
        #  ENHANCED: Fallback suggestions also include citations if available
        fallback_suggestions = []
        for rule in violated_rules[:3]:
            if isinstance(rule, dict):
                rule_text = rule.get('rule_text', str(rule))
                citation = rule.get('citation', '')
                
                suggestion = f"Review and fix: {rule_text}"
                if citation:
                    suggestion += f"\n   Source: {citation}"
                fallback_suggestions.append(suggestion)
            else:
                fallback_suggestions.append(f"Review and fix: {rule}")
        
        return fallback_suggestions if fallback_suggestions else [
            "Review and fix the critical rule violations listed above",
            "Check pipeline configuration against failed requirements",
            "Ensure all dependencies and tools are properly configured"
        ]
        
def generate_confidence_improvement_suggestions(rule_analysis: Dict, query: str, current_confidence: int) -> List[str]:
    """
    Generate specific suggestions to improve confidence based on actual analysis results
    """
    suggestions = []
    
    # Calculate confidence gap
    confidence_gap = 90 - current_confidence
    if confidence_gap <= 0:
        return []
    
    # Analyze violation types
    critical_count = len(rule_analysis.get('critical_violations', []))
    warning_count = len(rule_analysis.get('warnings', []))
    
    if critical_count > 0:
        potential_gain = min(confidence_gap, critical_count * 8)
        suggestions.append(f" Fix {critical_count} critical violations (+{potential_gain}% confidence)")
    
    if warning_count > 0:
        potential_gain = min(confidence_gap // 2, warning_count * 3)
        suggestions.append(f"⚠️ Address {warning_count} warning issues (+{potential_gain}% confidence)")
    
    # Get AI-generated specific suggestions
    ai_suggestions = generate_intelligent_suggestions_from_violations(rule_analysis, query)
    suggestions.extend(ai_suggestions[:3])
    
    return suggestions


def sync_get_rag_response(query: str) -> str:
    print("=" * 50)
    print("REAL YOCTO FILE ANALYSIS + VECTORSTORE")
    print("=" * 50)
    
    # Check if this is a pipeline analysis request
    is_pipeline_query = any(keyword in query.lower() for keyword in [
        'pipeline', 'stage', 'steps', 'bitbake', 'yocto', 'jenkins', 'build', 'workflow'
    ])
    
    if not is_pipeline_query:
        return "No pipeline detected - please provide a pipeline to analyze"
    
    # STEP 1: Retrieve relevant documents from vectorstore
    all_docs = vectorstore.similarity_search(query, k=20)
    print(f" Retrieved {len(all_docs)} documents from vectorstore")
    
    # STEP 2: Define confluence_docs and log_docs
    confluence_docs = []
    log_docs = []
    
    for doc in all_docs:
        source = doc.metadata.get('source', '')
        if any(keyword in source.lower() for keyword in ['static_knowledge', 'yocto', 'build', 'guideline', 'documentation']):
            confluence_docs.append(doc)
        elif any(keyword in source.lower() for keyword in ['dynamic_knowledge', 'log', 'jenkins_log', 'build_logs']):
            log_docs.append(doc)
        else:
            confluence_docs.append(doc)
    
    print(f" Processing {len(confluence_docs)} documentation docs and {len(log_docs)} log docs")
    
    # STEP 3: **REAL YOCTO WORKSPACE FILE ANALYSIS**
    print(" ANALYZING ACTUAL YOCTO WORKSPACE FILES...")
    
    # Try multiple possible workspace locations
    possible_workspaces = [
        "/var/jenkins_home/workspace/Yocto-Build-Pipeline",
        "/yocto-builds",
        "/var/jenkins_home/workspace/disk-space-check-analysis",
        os.getenv("WORKSPACE", "/tmp")
    ]
    
    yocto_file_paths = []
    workspace_files = []
    actual_workspace = None
    
    for workspace_dir in possible_workspaces:
        if os.path.exists(workspace_dir):
            actual_workspace = workspace_dir
            print(f"✅ Found workspace: {workspace_dir}")
            
            for root, _, files in os.walk(workspace_dir):
                for fn in files:
                    full_path = os.path.join(root, fn)
                    rel_path = os.path.relpath(full_path, workspace_dir)
                    
                    if (fn == "Jenkinsfile" or 
                        fn.endswith(('.bb', '.conf', '.bbappend', '.inc', '.bbclass')) or
                        fn in ('layer.conf', 'local.conf', 'bblayers.conf')):
                        workspace_files.append(rel_path)
                        
                        # Collect Yocto files for CONTENT analysis
                        if fn.endswith(('.bb', '.conf', '.bbappend', '.inc', '.bbclass')) or fn in ('layer.conf', 'local.conf', 'bblayers.conf'):
                            yocto_file_paths.append(full_path)
            break
    
    if not actual_workspace:
        print(" No workspace found - using generic analysis only")
    
    print(f" Found {len(workspace_files)} workspace files")
    print(f" Found {len(yocto_file_paths)} Yocto files for CONTENT analysis")
    
    # STEP 4: **EXTRACT YOCTO-SPECIFIC RULES FROM ACTUAL FILES**
    yocto_specific_rules = []
    if yocto_file_paths:
        print("🔍 Extracting rules from ACTUAL Yocto file contents...")
        yocto_specific_rules = extract_universal_yocto_rules(query)
        print(f"✅ Extracted {len(yocto_specific_rules)} REAL Yocto-specific rules")
        
        # Debug: Show actual Yocto rules found
        for rule in yocto_specific_rules[:3]:
            print(f"   📋 {rule['rule_type']}: {rule['rule_text'][:80]}...")


    # STEP 5: Extract rules from vectorstore documents WITH METADATA
    all_intelligent_rules = []
    for doc in confluence_docs[:10]:
        content = doc.page_content[:1200]
        doc_rules = extract_lightweight_intelligent_rules(content, 'documentation')
        
        #  NEW: Attach document metadata to each rule
        for rule in doc_rules:
            if isinstance(rule, dict):
                rule['metadata'] = doc.metadata  # ← This is the key fix!
        
        all_intelligent_rules.extend(doc_rules)

    for doc in log_docs[:5]:
        content = doc.page_content[:800]
        doc_rules = extract_lightweight_intelligent_rules(content, 'log')
        
        #  NEW: Attach document metadata to each rule
        for rule in doc_rules:
            if isinstance(rule, dict):
                rule['metadata'] = doc.metadata  # ← This is the key fix!
        
        all_intelligent_rules.extend(doc_rules)
    print("\n=== DEBUG: Checking if metadata has URLs ===")
    for i, rule in enumerate(all_intelligent_rules[:3]):
        if isinstance(rule, dict):
            meta = rule.get('metadata', {})
            print(f"Rule {i}: {rule.get('rule_text', '')[:50]}...")
            print(f"  Metadata keys: {list(meta.keys())}")
            print(f"  Confluence URL: {meta.get('confluence_url', 'MISSING!')}")
    print("=== END DEBUG ===\n")


    print(f"✅ Extracted {len(all_intelligent_rules)} intelligent rules WITH METADATA from vectorstore")

    
    # STEP 6: **COMBINE VECTORSTORE + REAL YOCTO RULES**
    combined_rules = all_intelligent_rules.copy() 
    for rule in yocto_specific_rules:
        combined_rules.append({
            'rule_text': rule['rule_text'],
            'rule_type': rule['rule_type'],
            'confidence': rule.get('confidence', 0.9),
            'source': f"YOCTO_FILE:{rule.get('source', 'workspace')}",
            'metadata': {}  # No doc metadata for workspace files
        })
    combined_rules = enhance_rules_with_citations(combined_rules)
    print(f"✅ Enhanced {len(combined_rules)} rules with citations")
    print(f" Combined total: {len(combined_rules)} rules ({len(all_intelligent_rules)} vectorstore + {len(yocto_specific_rules)} REAL Yocto)")
    
    # STEP 7: **FIXED ANALYSIS** - Analyze pipeline against combined rules
    rule_analysis = analyze_pipeline_with_adaptive_intelligence(query, combined_rules)
    
    # STEP 8: **FIXED CONFIDENCE CALCULATION** - Based on actual math
    final_confidence, prediction = calculate_proper_confidence_and_prediction(rule_analysis, yocto_specific_rules)
    
    # STEP 9: REAL AI SUGGESTIONS - Based on failed rules  
    print(f" Checking AI suggestions: confidence={final_confidence}%, violations={len(rule_analysis.get('violated_rules_details', []))}")

    if final_confidence < 90:
        if rule_analysis and rule_analysis.get('violated_rules_details'):
            print(f" Generating 100% AI-driven ultra-specific suggestions...")
            ai_suggestions = generate_ultra_specific_suggestions_from_failed_rules(
                    rule_analysis, query, actual_workspace
            )
        else:
            print(" No violated rule details found for suggestion generation")
            ai_suggestions = []
    else:
        ai_suggestions = []

    print(f" Final AI suggestions: {len(ai_suggestions)} generated")

    
    # STEP 10: Get historical patterns
    try:
        success_patterns, failure_patterns = get_historical_patterns_from_logs()
    except:
        success_patterns = ['Build completed successfully', 'Dependencies resolved properly']
        failure_patterns = ['Timeout or resource issues', 'Configuration problems']
    
    # STEP 11: **BUILD COMPREHENSIVE RESPONSE WITH CORRECT VALUES**
    detected_technologies = ['yocto', 'jenkins', 'git'] if yocto_specific_rules else ['jenkins', 'general']
    
    response = f"""DETECTED_STACK: {', '.join(detected_technologies)}
ESTABLISHED_RULES:
"""
    
    # Show actual rules found
    for rule_text in (rule_analysis.get('evaluated_rules', []) if rule_analysis else [])[:25]:
        response += f"• {rule_text}\n"
    
    # **REAL YOCTO WORKSPACE ANALYSIS SECTION**
    if yocto_specific_rules:
        yocto_dependencies = [r for r in yocto_specific_rules if 'DEPENDENCY' in r['rule_type']]
        yocto_configs = [r for r in yocto_specific_rules if 'CONFIG' in r['rule_type'] or 'THREADS' in r['rule_type'] or 'MACHINE' in r['rule_type']]
        yocto_sources = [r for r in yocto_specific_rules if 'SOURCE' in r['rule_type']]
        
        response += f"""
 REAL YOCTO WORKSPACE ANALYSIS:
Found {len(yocto_specific_rules)} specific rules from {len(yocto_file_paths)} actual Yocto files:
• Dependencies: {len(yocto_dependencies)}
• Configurations: {len(yocto_configs)}
• Sources: {len(yocto_sources)}

ACTUAL YOCTO FILES ANALYZED:
{chr(10).join([f"• {os.path.basename(f)}" for f in yocto_file_paths[:8]])}

YOCTO-SPECIFIC RULES FROM FILES:
{chr(10).join([f"• {r['rule_text'][:100]}" for r in yocto_specific_rules[:8]])}
"""
    else:
        response += f"""
⚠️ YOCTO ANALYSIS WARNING:
No Yocto files found in workspace locations:
{chr(10).join([f"• {ws}" for ws in possible_workspaces[:3]])}
"""
    
    response += f"""HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
✅ {success_patterns[0] if success_patterns else 'Build completed successfully'}
✅ {success_patterns[1] if len(success_patterns) > 1 else 'Dependencies resolved properly'}
FAILURE PATTERNS:
❌ {failure_patterns[0] if failure_patterns else 'Timeout or resource issues'}
❌ {failure_patterns[1] if len(failure_patterns) > 1 else 'Configuration problems'}
APPLICABLE_RULES: {rule_analysis.get('total_rules', len(combined_rules)) if rule_analysis else len(combined_rules)}
SATISFIED_RULES: {rule_analysis.get('satisfied_count', 0) if rule_analysis else 0}
VIOLATED_RULES: {rule_analysis.get('violated_count', 0) if rule_analysis else 0}
RISK_FACTORS: {', '.join(rule_analysis.get('critical_violations', [])[:3]) if rule_analysis and rule_analysis.get('critical_violations') else 'None detected'}
PREDICTION: {prediction}
CONFIDENCE: {final_confidence}%
REASONING: Enhanced analysis using {len(all_intelligent_rules)} vectorstore rules + {len(yocto_specific_rules)} REAL Yocto workspace file rules with CORRECT mathematical calculations.
"""

    # **REAL AI-GENERATED SUGGESTIONS**
    if ai_suggestions:
        response += f"""
 AI-GENERATED SUGGESTIONS TO REACH 90%+ CONFIDENCE:
{chr(10).join([f"• {suggestion}" for suggestion in ai_suggestions])}

 ANALYSIS SUMMARY:
• Critical Issues: {len(rule_analysis.get('critical_violations', []))}
• Warning Issues: {len(rule_analysis.get('warnings', []))}
• Confidence Gap: {90 - final_confidence}%
• Violated Rules: {rule_analysis.get('violated_count', 0)}/{rule_analysis.get('total_rules', len(combined_rules))}


"""
    elif final_confidence >= 90:
        response += f"""
✅ EXCELLENT CONFIDENCE LEVEL!
Pipeline analysis shows high probability of success.
All critical requirements are satisfied.
"""
    
    return response

def extract_basic_rules_fallback(content: str) -> List[Dict]:
    """Fallback rule extraction if intelligent extraction fails"""
    rules = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
            
        # Basic requirement patterns
        if any(word in line.lower() for word in ['must', 'should', 'required', 'ensure']):
            rules.append({
                'rule_text': line,
                'rule_type': 'BASIC_REQUIREMENT',
                'confidence': 0.6,
                'source': 'FALLBACK_EXTRACTION'
            })
    
    return rules

def prioritize_rules_by_relevance(rules: List[Dict], query: str) -> List[Dict]:
    """Prioritize rules based on relevance to the query"""
    query_lower = query.lower()
    
    for rule in rules:
        rule_text_lower = rule['rule_text'].lower()
        
        # Calculate relevance score
        relevance_score = 0
        
        # Technology relevance
        tech_keywords = ['yocto', 'bitbake', 'jenkins', 'docker', 'kubernetes', 'build', 'test', 'deploy']
        relevance_score += sum(1 for keyword in tech_keywords if keyword in rule_text_lower and keyword in query_lower) * 0.3
        
        # Action relevance
        action_keywords = ['configure', 'setup', 'install', 'build', 'test', 'deploy', 'verify']
        relevance_score += sum(1 for keyword in action_keywords if keyword in rule_text_lower) * 0.2
        
        # Requirement level boost
        if rule.get('rule_type') == 'MANDATORY':
            relevance_score += 0.4
        elif rule.get('rule_type') == 'RECOMMENDED':
            relevance_score += 0.2
        
        # Update confidence based on relevance
        rule['relevance_score'] = relevance_score
        rule['confidence'] = min(1.0, rule.get('confidence', 0.5) + relevance_score)
    
    # Sort by relevance and confidence
    return sorted(rules, key=lambda x: (x.get('relevance_score', 0), x.get('confidence', 0.5)), reverse=True)

def remove_duplicate_rules(rules: List[Dict]) -> List[Dict]:
    """Remove duplicate rules while preserving the highest confidence version"""
    seen_rules = {}
    
    for rule in rules:
        rule_key = rule['rule_text'].lower().strip()[:100]  # Use first 100 chars as key
        
        if rule_key not in seen_rules:
            seen_rules[rule_key] = rule
        elif rule.get('confidence', 0) > seen_rules[rule_key].get('confidence', 0):
            seen_rules[rule_key] = rule
    
    return list(seen_rules.values())


def analyze_pipeline_workspace_dynamically(pipeline_content: str) -> Dict:
    """
    GENERIC workspace analysis that works for ANY pipeline from ANY platform
    Extracts file paths, technologies, and configurations from pipeline content itself
    """
    analysis = {
        'workspace_files': [],
        'detected_technologies': [],
        'configuration_files': [],
        'build_files': [],
        'workspace_paths': [],
        'pipeline_specific_rules': []
    }
    
    print("🔍 Performing GENERIC pipeline analysis...")
    
    # 1. EXTRACT FILE PATHS FROM PIPELINE CONTENT (ANY CI/CD PLATFORM)
    file_path_patterns = [
        r'["\']([^"\']+\.(bb|conf|bbappend|inc|bbclass|json|yaml|yml|xml|properties|gradle|pom|dockerfile))["\']',
        r'(?:file|path|dir|directory)[:\s=]+["\']([^"\']+)["\']',
        r'(?:checkout|clone|copy|cp|mv)\s+["\']?([^\s"\']+)["\']?',
        r'(?:cd|pushd|popd)\s+([^\s\n]+)',
        r'(?:source|\.|\s)([^\s]+\.(?:sh|bat|cmd|ps1))',
        r'(?:find|ls|cat|head|tail)\s+([^\s\n]+)',
    ]
    
    for pattern in file_path_patterns:
        matches = re.findall(pattern, pipeline_content, re.IGNORECASE)
        for match in matches:
            file_path = match[0] if isinstance(match, tuple) else match
            if file_path and len(file_path) < 200:
                analysis['workspace_files'].append(file_path)
    
    # 2. DETECT TECHNOLOGIES FROM PIPELINE CONTENT
    tech_patterns = {
        'yocto': [r'\bbitbake\b', r'\byocto\b', r'\.bb\b', r'\.bbappend\b', r'\bmeta-\w+'],
        'docker': [r'\bdocker\b', r'\bdockerfile\b', r'docker\s+build', r'docker\s+run'],
        'maven': [r'\bmaven\b', r'\bmvn\b', r'pom\.xml'],
        'gradle': [r'\bgradle\b', r'\bgradlew\b', r'build\.gradle'],
        'node': [r'\bnpm\b', r'\byarn\b', r'package\.json', r'node_modules'],
        'python': [r'\bpip\b', r'\bconda\b', r'requirements\.txt', r'setup\.py'],
        'git': [r'\bgit\b', r'git\s+clone', r'git\s+checkout'],
        'jenkins': [r'\bjenkins\b', r'\bJenkinsfile\b', r'pipeline\s*{'],
        'kubernetes': [r'\bkubectl\b', r'\bkubernetes\b', r'\.yaml\b.*deploy'],
        'terraform': [r'\bterraform\b', r'\.tf\b'],
        'ansible': [r'\bansible\b', r'ansible-playbook', r'\.yml\b.*play']
    }
    
    for tech, patterns in tech_patterns.items():
        if any(re.search(pattern, pipeline_content, re.IGNORECASE) for pattern in patterns):
            analysis['detected_technologies'].append(tech)
    
    # 3. EXTRACT CONFIGURATION VALUES FROM PIPELINE
    config_patterns = [
        r'([A-Z_]{3,})\s*[=:]\s*["\']?([^"\'\n]{1,100})["\']?',  # ENV_VAR = value
        r'export\s+([A-Z_]+)=([^\s\n]+)',  # export VAR=value
        r'set\s+([A-Z_]+)=([^\s\n]+)',     # set VAR=value
        r'([a-zA-Z_]+)\s*[:=]\s*(\d+[a-zA-Z]*)',  # timeout: 6hours
        r'(machine|distro|parallel_make|bb_number_threads)\s*[=:]\s*["\']?([^"\'\n]+)["\']?',
    ]
    
    for pattern in config_patterns:
        matches = re.findall(pattern, pipeline_content, re.IGNORECASE)
        for key, value in matches:
            analysis['configuration_files'].append(f"{key.upper()}: {value}")
    
    # 4. EXTRACT BUILD COMMANDS AND STAGES
    build_command_patterns = [
        r'(bitbake\s+[^\n]+)',
        r'(docker\s+build[^\n]+)',
        r'(mvn\s+[^\n]+)',
        r'(gradle\s+[^\n]+)',
        r'(npm\s+[^\n]+)',
        r'(make\s+[^\n]+)',
        r'(cmake\s+[^\n]+)'
    ]
    
    for pattern in build_command_patterns:
        matches = re.findall(pattern, pipeline_content, re.IGNORECASE)
        analysis['build_files'].extend(matches)
    
    # 5. EXTRACT WORKSPACE PATHS FROM PIPELINE
    workspace_patterns = [
        r'(?:workspace|workdir|working.?directory)[:\s=]+["\']?([^"\'\n]+)["\']?',
        r'(?:cd|chdir)\s+([^\s\n]+)',
        r'WORKDIR\s+([^\n]+)',
        r'(?:pwd|cwd)[:\s=]+([^\s\n]+)'
    ]
    
    for pattern in workspace_patterns:
        matches = re.findall(pattern, pipeline_content, re.IGNORECASE)
        analysis['workspace_paths'].extend(matches)
    
    print(f"✅ GENERIC analysis extracted:")
    print(f"  📁 Files: {len(analysis['workspace_files'])} detected")
    print(f"  🔧 Technologies: {analysis['detected_technologies']}")
    print(f"  ⚙️ Configurations: {len(analysis['configuration_files'])} detected")
    print(f"  🏗️ Build commands: {len(analysis['build_files'])} detected")
    
    return analysis

def extract_generic_pipeline_rules(pipeline_analysis: Dict) -> List[Dict]:
    """
    Extract technology-specific rules based on detected technologies
    NO hardcoded paths - everything derived from pipeline content
    """
    rules = []
    detected_tech = pipeline_analysis['detected_technologies']
    
    print(f"🧠 Extracting rules for detected technologies: {detected_tech}")
    
    # YOCTO-SPECIFIC RULES (only if Yocto detected)
    if 'yocto' in detected_tech:
        # Check if essential Yocto configurations are present
        configs = ' '.join(pipeline_analysis['configuration_files']).lower()
        
        if 'machine' not in configs:
            rules.append({
                'rule_text': 'Yocto build requires MACHINE configuration (e.g., qemux86-64)',
                'rule_type': 'YOCTO_MACHINE_MISSING',
                'confidence': 0.90,
                'source': 'GENERIC_YOCTO_ANALYSIS'
            })
        
        if 'bb_number_threads' not in configs:
            rules.append({
                'rule_text': 'Yocto build should configure BB_NUMBER_THREADS for performance',
                'rule_type': 'YOCTO_THREADS_MISSING', 
                'confidence': 0.80,
                'source': 'GENERIC_YOCTO_ANALYSIS'
            })
        
        if 'bitbake' in ' '.join(pipeline_analysis['build_files']).lower():
            rules.append({
                'rule_text': 'BitBake command detected - ensure proper Yocto environment setup',
                'rule_type': 'YOCTO_BITBAKE_DETECTED',
                'confidence': 0.95,
                'source': 'GENERIC_YOCTO_ANALYSIS'
            })
    
    # DOCKER-SPECIFIC RULES
    if 'docker' in detected_tech:
        rules.append({
            'rule_text': 'Docker build requires sufficient disk space and daemon access',
            'rule_type': 'DOCKER_REQUIREMENTS',
            'confidence': 0.85,
            'source': 'GENERIC_DOCKER_ANALYSIS'
        })
    
    # GIT-SPECIFIC RULES
    if 'git' in detected_tech:
        rules.append({
            'rule_text': 'Git operations require network connectivity and authentication',
            'rule_type': 'GIT_REQUIREMENTS',
            'confidence': 0.85,
            'source': 'GENERIC_GIT_ANALYSIS'
        })
    
    # JENKINS-SPECIFIC RULES
    if 'jenkins' in detected_tech:
        pipeline_content = ' '.join(pipeline_analysis['build_files'] + pipeline_analysis['configuration_files'])
        
        if 'timeout' not in pipeline_content.lower():
            rules.append({
                'rule_text': 'Jenkins pipeline should configure timeout to prevent hanging builds',
                'rule_type': 'JENKINS_TIMEOUT_MISSING',
                'confidence': 0.85,
                'source': 'GENERIC_JENKINS_ANALYSIS'
            })
        
        if 'agent' not in pipeline_content.lower():
            rules.append({
                'rule_text': 'Jenkins pipeline should specify appropriate build agent',
                'rule_type': 'JENKINS_AGENT_MISSING',
                'confidence': 0.80,
                'source': 'GENERIC_JENKINS_ANALYSIS'
            })
    
    print(f"✅ Extracted {len(rules)} technology-specific rules")
    return rules



def analyze_pipeline_with_adaptive_intelligence(pipeline_text: str, intelligent_rules: List[Dict]) -> Dict:
    """
    FIXED: Proper pipeline analysis with violated rule tracking
    """
    pipeline_lower = pipeline_text.lower()
    
    print(f" ANALYZING {len(intelligent_rules)} rules with proper tracking...")
    
    analysis = {
        'evaluated_rules': [],
        'satisfied_count': 0,
        'violated_count': 0,
        'total_rules': 0,
        'critical_violations': [],
        'warnings': [],
        'risk_factors': [],
        'violated_rules_details': []  # CRITICAL: Store failed rules for AI
    }
    
    # Filter and deduplicate rules
    filtered_rules = []
    for rule in intelligent_rules:
        rule_text = rule.get('rule_text', '').strip()
        
        # Skip meaningless fragment rules
        if any(fragment in rule_text.lower() for fragment in [
            'location', 'contents', 'pipeline {', 'echo', '+ echo', '# ', 
            '[pipeline]', 'writeFile', 'archiveArtifacts'
        ]):
            continue
            
        if len(rule_text) < 15:  # Skip very short rules
            continue
            
        if not any(meaningful in rule_text.lower() for meaningful in [
            'build', 'recipe', 'machine', 'bitbake', 'yocto', 'environment', 
            'source', 'dependency', 'layer', 'image', 'configure', 'install'
        ]):
            continue
            
        filtered_rules.append(rule)
    
    analysis['total_rules'] = len(filtered_rules)
    print(f" Filtered to {analysis['total_rules']} meaningful rules")
    
    # Evaluate each rule
    for rule in filtered_rules:
        rule_text = rule.get('rule_text', '')
        rule_type = rule.get('rule_type', 'UNKNOWN')
        confidence = rule.get('confidence', 0.7)
        source = rule.get('source', 'UNKNOWN')
        
        # Evaluate rule against pipeline
        rule_satisfied = evaluate_rule_against_pipeline_adaptive(rule, pipeline_text, pipeline_lower)
        
        # Format for display
        status = "PASS" if rule_satisfied else "FAIL"
        confidence_indicator = f" (conf: {confidence:.1f})" if confidence < 0.8 else ""
        
        formatted_rule = f"{rule_text[:80]}... - {status}{confidence_indicator}"
        analysis['evaluated_rules'].append(formatted_rule)
        
        # PROPERLY COUNT AND TRACK VIOLATIONS
        if rule_satisfied:
            analysis['satisfied_count'] += 1
        else:
            analysis['violated_count'] += 1
            
            # *** KEY FIX: Store violated rule details for AI suggestions ***
            analysis['violated_rules_details'].append({
                'rule_text': rule_text,
                'rule_type': rule_type,
                'confidence': confidence,
                'source': source
            })
            
            print(f"🚫 FAILED RULE: {rule_text[:60]}...")
            
            # Categorize violations
            if any(critical in rule_text.lower() for critical in ['must', 'critical', 'essential', 'mandatory']):
                analysis['critical_violations'].append(rule_text)
                analysis['risk_factors'].append(f"{rule_type}: {rule_text[:50]}...")
            else:
                analysis['warnings'].append(rule_text)
    
    print(f"✅ Analysis complete: {analysis['satisfied_count']} PASS, {analysis['violated_count']} FAIL")
    print(f" Violations stored for AI: {len(analysis['violated_rules_details'])} rules")
    
    return analysis

def calculate_proper_confidence_and_prediction(rule_analysis: Dict, yocto_specific_rules: List[Dict]) -> Tuple[int, str]:
    """
    FIXED: Correct confidence calculation and prediction
    """
    total_rules = rule_analysis.get('total_rules', 1)
    satisfied_rules = rule_analysis.get('satisfied_count', 0)
    violated_rules = rule_analysis.get('violated_count', 0)
    critical_violations = len(rule_analysis.get('critical_violations', []))
    
    if total_rules == 0:
        return 50, "UNKNOWN"
    
    # CORRECT MATH: Base confidence = pass rate
    pass_rate = (satisfied_rules / total_rules) * 100
    print(f" MATH CHECK: {satisfied_rules}/{total_rules} = {pass_rate:.1f}%")
    
    # Apply penalties
    critical_penalty = critical_violations * 10  # 10% per critical issue
    base_confidence = pass_rate - critical_penalty
    
    # Minor Yocto adjustments
    yocto_bonus = 0
    if yocto_specific_rules:
        if any('MACHINE' in r.get('rule_type', '') for r in yocto_specific_rules):
            yocto_bonus += 2
        if any('ENVIRONMENT' in r.get('rule_type', '') for r in yocto_specific_rules):
            yocto_bonus += 3
    
    final_confidence = max(15, min(95, int(base_confidence + yocto_bonus)))
    
    # CORRECT PREDICTION LOGIC
    if final_confidence >= 75:
        prediction = "PASS"
    elif final_confidence >= 50:
        prediction = "HIGH-RISK" 
    else:
        prediction = "FAIL"
    
    print(f" FINAL: {pass_rate:.1f}% - {critical_penalty}% + {yocto_bonus}% = {final_confidence}% → {prediction}")
    
    return final_confidence, prediction


def generate_ultra_specific_suggestions_from_failed_rules(rule_analysis: Dict, pipeline_text: str, workspace_path: str) -> List[str]:
    """
    FIXED VERSION: Corrects file scanning and AI parsing issues
    """
    import time
    
    start_time = time.time()
    
    violated_rules_details = rule_analysis.get('violated_rules_details', [])
    print(f"🔍 FIXED: Processing {len(violated_rules_details)} violated rules")
    
    if not violated_rules_details:
        return []
    
    # STEP 1: FIXED workspace file scanning
    found_files = scan_workspace_files_fixed(workspace_path)
    print(f"📁 FIXED: Found {len(found_files)} relevant files:")
    for file_path in found_files[:5]:
        print(f"   - {file_path}")
    
    # STEP 2: Extract specific file details
    specific_details = extract_specific_file_details_fixed(violated_rules_details)
    print(f"🔍 FIXED: Looking for specific files: {specific_details['bbappend_files']}")
    
    # STEP 3: Match found files to failed rules
    matched_files = match_files_to_failures(specific_details, found_files)
    print(f"🎯 FIXED: Matched files: {matched_files}")
    
    # STEP 4: Generate AI suggestions with FIXED parsing
    try:
        suggestions = generate_fixed_ai_suggestions(
            violated_rules_details,
            specific_details,
            matched_files,
            workspace_path
        )
        
        elapsed = time.time() - start_time
        print(f"✅ FIXED: Generated {len(suggestions)} suggestions in {elapsed:.2f}s")
        
        return suggestions
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FIXED: Failed after {elapsed:.2f}s: {e}")
        return generate_immediate_specific_suggestions(specific_details, matched_files)

def scan_workspace_files_fixed(workspace_path: str) -> List[str]:
    """
    FIXED: Properly scan workspace files without overly restrictive limits
    """
    found_files = []
    
    if not workspace_path or not os.path.exists(workspace_path):
        return found_files
    
    try:
        files_scanned = 0
        max_files = 200  # Increased limit
        
        for root, dirs, files in os.walk(workspace_path):
            # LESS restrictive depth limit
            depth = root.replace(workspace_path, '').count(os.sep)
            if depth > 5:  # Increased from 2 to 5
                continue
            
            # Skip fewer directories
            dirs[:] = [d for d in dirs if d not in ['tmp', 'sstate-cache', 'downloads', '.git']]
            
            for file_name in files:
                if files_scanned >= max_files:
                    break
                
                file_path = os.path.join(root, file_name)
                
                # BROADER file filtering
                if should_include_file_fixed(file_name, file_path):
                    rel_path = os.path.relpath(file_path, workspace_path)
                    found_files.append(rel_path)
                
                files_scanned += 1
                
            if files_scanned >= max_files:
                break
    
    except Exception as e:
        print(f"⚠️ FIXED: File scan error: {e}")
    
    return found_files

def should_include_file_fixed(file_name: str, file_path: str) -> bool:
    """
    FIXED: More inclusive file filtering
    """
    try:
        # Skip very large files only
        if os.path.getsize(file_path) > 500000:  # 500KB limit (was 100KB)
            return False
    except:
        return False
    
    # Include more file types
    relevant_extensions = {'.bbappend', '.conf', '.bb', '.bbclass', '.inc', '.py'}
    relevant_names = {'local.conf', 'bblayers.conf', 'layer.conf', 'Jenkinsfile'}
    
    return (any(file_name.endswith(ext) for ext in relevant_extensions) or 
            file_name in relevant_names or
            (not '.' in file_name and 'conf' in file_name.lower()))

def extract_specific_file_details_fixed(violated_rules: List[Dict]) -> Dict:
    """
    FIXED: Extract specific file details from rules
    """
    details = {
        'bbappend_files': [],
        'config_files': [],
        'environment_issues': [],
        'disk_issues': []
    }
    
    for rule in violated_rules:
        rule_text = rule.get('rule_text', '').lower()
        rule_type = rule.get('rule_type', '')
        
        # Extract .bbappend files more accurately
        if 'bbappend' in rule_text:
            # Look for specific file patterns
            bbappend_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bbappend)', rule.get('rule_text', ''))
            details['bbappend_files'].extend(bbappend_matches)
        
        # Recipe modification rules
        if 'recipe modification' in rule_text:
            recipe_matches = re.findall(r'([a-zA-Z0-9_-]+(?:[_-][\d\.]+)?\.bbappend)', rule.get('rule_text', ''))
            details['bbappend_files'].extend(recipe_matches)
        
        # Environment issues
        if 'environment' in rule_text or 'YOCTO_UNIVERSAL_ENV' in rule_type:
            details['environment_issues'].append('BitBake environment initialization')
        
        # Disk space issues
        if 'disk space' in rule_text or 'YOCTO_UNIVERSAL_DISK' in rule_type:
            details['disk_issues'].append('Insufficient disk space')
        
        # Config files
        if 'local.conf' in rule_text:
            details['config_files'].append('local.conf')
        if 'bblayers.conf' in rule_text:
            details['config_files'].append('bblayers.conf')
    
    # Remove duplicates
    details['bbappend_files'] = list(set(details['bbappend_files']))
    details['config_files'] = list(set(details['config_files']))
    
    return details

def match_files_to_failures(specific_details: Dict, found_files: List[str]) -> Dict:
    """
    FIXED: Match found files to specific failures
    """
    matched = {
        'existing_bbappend': {},
        'missing_bbappend': [],
        'existing_config': {},
        'missing_config': []
    }
    
    # Match .bbappend files
    for bbappend_file in specific_details['bbappend_files']:
        matching_files = [f for f in found_files if bbappend_file in f or os.path.basename(f) == bbappend_file]
        
        if matching_files:
            matched['existing_bbappend'][bbappend_file] = matching_files[0]
        else:
            matched['missing_bbappend'].append(bbappend_file)
    
    # Match config files
    for config_file in specific_details['config_files']:
        matching_files = [f for f in found_files if config_file in f]
        
        if matching_files:
            matched['existing_config'][config_file] = matching_files[0]
        else:
            matched['missing_config'].append(config_file)
    
    return matched

def generate_fixed_ai_suggestions(violated_rules: List[Dict], specific_details: Dict, matched_files: Dict, workspace_path: str) -> List[str]:
    """
    100% AI-DRIVEN: No hardcoded solutions - works with any Yocto pipeline globally
    """
    try:
        # Build COMPREHENSIVE context from actual analysis
        ai_prompt = build_universal_ai_prompt(violated_rules, specific_details, matched_files, workspace_path)
        
        print("🤖 UNIVERSAL: Calling AI with comprehensive analysis...")
        ai_response = llm.invoke(ai_prompt)
        
        print(f"🤖 UNIVERSAL: AI response length: {len(ai_response)} characters")
        print(f"🤖 UNIVERSAL: AI response preview:\n{ai_response[:300]}...")
        
        # Parse complete AI response
        suggestions = parse_universal_ai_response(ai_response)
        
        print(f"🤖 UNIVERSAL: Parsed {len(suggestions)} complete suggestions")
        
        if suggestions:
            return suggestions[:5]
        else:
            print("🤖 UNIVERSAL: No complete suggestions parsed, trying AI-driven fallback...")
            return generate_ai_driven_fallback(violated_rules, specific_details, matched_files, workspace_path)
    
    except Exception as e:
        print(f"❌ UNIVERSAL: AI call failed: {e}")
        return generate_ai_driven_fallback(violated_rules, specific_details, matched_files, workspace_path)

def build_universal_ai_prompt(violated_rules: List[Dict], specific_details: Dict, matched_files: Dict, workspace_path: str) -> str:
    """
    Build comprehensive AI prompt with ALL context - no assumptions
    """
    prompt_sections = []
    
    # Section 1: Universal context
    prompt_sections.append("You are a world-class Yocto/OpenEmbedded expert. Analyze this SPECIFIC build failure and provide COMPLETE, actionable solutions.")
    
    # Section 2: Actual failed rules with full context
    prompt_sections.append("\nSPECIFIC BUILD FAILURES TO FIX:")
    for i, rule in enumerate(violated_rules, 1):
        rule_context = f"""
FAILURE {i}:
- Rule: {rule.get('rule_text', 'Unknown')}
- Type: {rule.get('rule_type', 'Unknown')}
- Source: {rule.get('source', 'Unknown')}
- Confidence: {rule.get('confidence', 0.0)}"""
        prompt_sections.append(rule_context)
    
    # Section 3: Workspace analysis results
    prompt_sections.append(f"\nWORKSPACE ANALYSIS RESULTS:")
    prompt_sections.append(f"- Workspace Path: {workspace_path}")
    prompt_sections.append(f"- Files Mentioned in Failures: {specific_details.get('bbappend_files', [])}")
    prompt_sections.append(f"- Config Files Mentioned: {specific_details.get('config_files', [])}")
    prompt_sections.append(f"- Environment Issues: {specific_details.get('environment_issues', [])}")
    prompt_sections.append(f"- Disk Issues: {specific_details.get('disk_issues', [])}")
    
    # Section 4: File existence analysis
    prompt_sections.append(f"\nFILE EXISTENCE ANALYSIS:")
    if matched_files['existing_bbappend']:
        prompt_sections.append("EXISTING FILES (need fixes):")
        for file_name, file_path in matched_files['existing_bbappend'].items():
            prompt_sections.append(f"- {file_name} EXISTS at: {file_path}")
    
    if matched_files['missing_bbappend']:
        prompt_sections.append("MISSING FILES (need creation):")
        for file_name in matched_files['missing_bbappend']:
            prompt_sections.append(f"- {file_name} is MISSING")
    
    if matched_files['existing_config']:
        prompt_sections.append("EXISTING CONFIG FILES:")
        for file_name, file_path in matched_files['existing_config'].items():
            prompt_sections.append(f"- {file_name} EXISTS at: {file_path}")
    
    if matched_files['missing_config']:
        prompt_sections.append("MISSING CONFIG FILES:")
        for file_name in matched_files['missing_config']:
            prompt_sections.append(f"- {file_name} is MISSING")
    
    # Section 5: Requirements for AI response
    requirements = """
YOUR TASK - PROVIDE COMPLETE, UNIVERSAL SOLUTIONS:

For EACH failure above, analyze the specific issue and provide a COMPLETE solution that includes:
1. Exact action to take (create, edit, fix, configure)
2. Specific file paths (use actual workspace structure)
3. Complete file content or exact commands
4. Universal applicability (works for any Yocto project)

FORMAT REQUIREMENTS:
- Each solution starts with •
- Include complete code/content (not just descriptions)
- End with (+X% confidence)
- Be specific enough that any developer can copy-paste and implement
- Use the actual workspace path and file structure provided above

EXAMPLE OF COMPLETE SOLUTION FORMAT:
• Create missing recipe.bbappend: Create file [exact-path] with complete content:
[complete file content here]
(+X% confidence)

• Fix environment issue: Add the following to your pipeline:
[complete command or code here]
(+X% confidence)

ANALYZE THE SPECIFIC FAILURES AND WORKSPACE ABOVE - PROVIDE 5 COMPLETE SOLUTIONS:"""
    
    prompt_sections.append(requirements)
    
    return "\n".join(prompt_sections)

def parse_universal_ai_response(ai_response: str) -> List[str]:
    """
    Parse AI response universally - no assumptions about format
    """
    suggestions = []
    
    # Try multiple parsing strategies
    suggestions.extend(parse_bullet_format(ai_response))
    
    if not suggestions:
        suggestions.extend(parse_numbered_format(ai_response))
    
    if not suggestions:
        suggestions.extend(parse_any_confidence_format(ai_response))
    
    return suggestions

def parse_bullet_format(ai_response: str) -> List[str]:
    """
    Parse bullet point format responses
    """
    suggestions = []
    lines = ai_response.split('\n')
    current_suggestion = ""
    
    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('•'):
            # Save previous suggestion
            if current_suggestion:
                clean_suggestion = clean_universal_suggestion(current_suggestion)
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
            
            # Start new suggestion
            current_suggestion = line_stripped[1:].strip()
        
        elif current_suggestion and line_stripped:
            # Continue current suggestion
            current_suggestion += f"\n{line}"
        
        elif current_suggestion and not line_stripped:
            # Potential end of suggestion
            if has_confidence_indicator(current_suggestion):
                clean_suggestion = clean_universal_suggestion(current_suggestion)
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
                current_suggestion = ""
    
    # Handle last suggestion
    if current_suggestion:
        clean_suggestion = clean_universal_suggestion(current_suggestion)
        if clean_suggestion:
            suggestions.append(clean_suggestion)
    
    return suggestions

def parse_numbered_format(ai_response: str) -> List[str]:
    """
    Parse numbered list format responses
    """
    suggestions = []
    lines = ai_response.split('\n')
    current_suggestion = ""
    
    for line in lines:
        line_stripped = line.strip()
        
        if re.match(r'^\d+[\.\)]\s+', line_stripped):
            # Save previous suggestion
            if current_suggestion:
                clean_suggestion = clean_universal_suggestion(current_suggestion)
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
            
            # Start new suggestion
            current_suggestion = re.sub(r'^\d+[\.\)]\s+', '', line_stripped)
        
        elif current_suggestion and line_stripped:
            current_suggestion += f"\n{line}"
        
        elif current_suggestion and not line_stripped:
            if has_confidence_indicator(current_suggestion):
                clean_suggestion = clean_universal_suggestion(current_suggestion)
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
                current_suggestion = ""
    
    # Handle last suggestion
    if current_suggestion:
        clean_suggestion = clean_universal_suggestion(current_suggestion)
        if clean_suggestion:
            suggestions.append(clean_suggestion)
    
    return suggestions

def parse_any_confidence_format(ai_response: str) -> List[str]:
    """
    Parse any format that has confidence indicators
    """
    suggestions = []
    
    # Split by confidence patterns
    confidence_pattern = r'\(\+\d+%\s*confidence\)'
    parts = re.split(confidence_pattern, ai_response)
    
    for i in range(len(parts) - 1):
        suggestion_text = parts[i].strip()
        if suggestion_text and len(suggestion_text) > 50:
            # Find the confidence for this part
            confidence_match = re.search(confidence_pattern, ai_response[ai_response.find(suggestion_text) + len(suggestion_text):])
            if confidence_match:
                full_suggestion = f"{suggestion_text} {confidence_match.group()}"
                clean_suggestion = clean_universal_suggestion(full_suggestion)
                if clean_suggestion:
                    suggestions.append(clean_suggestion)
    
    return suggestions

def has_confidence_indicator(text: str) -> bool:
    """
    Check if text has confidence indicator
    """
    return bool(re.search(r'\(\+\d+%.*confidence\)', text, re.IGNORECASE))

def clean_universal_suggestion(suggestion_text: str) -> str:
    """
    Clean suggestion text universally - no hardcoded assumptions
    """
    if not suggestion_text or len(suggestion_text.strip()) < 20:
        return ""
    
    # Remove excessive whitespace while preserving structure
    lines = suggestion_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Ensure confidence indicator exists
    if not has_confidence_indicator(cleaned):
        cleaned += " (+15% confidence)"
    
    # Reasonable length limit
    if len(cleaned) > 1000:
        # Find a good breaking point
        confidence_match = re.search(r'\(\+\d+%.*confidence\)', cleaned[-200:])
        if confidence_match:
            break_point = cleaned.rfind(confidence_match.group())
            cleaned = cleaned[:break_point] + confidence_match.group()
        else:
            cleaned = cleaned[:1000] + " (+15% confidence)"
    
    return cleaned

def generate_ai_driven_fallback(violated_rules: List[Dict], specific_details: Dict, matched_files: Dict, workspace_path: str) -> List[str]:
    """
    AI-driven fallback when main AI fails - still no hardcoding
    """
    try:
        # Simplified AI prompt for fallback
        fallback_prompt = f"""Yocto build expert: Quick fixes needed for these failures:

FAILURES:
{chr(10).join([f"- {rule.get('rule_text', '')}" for rule in violated_rules[:3]])}

WORKSPACE: {workspace_path}
MISSING FILES: {matched_files.get('missing_bbappend', [])}
EXISTING FILES: {list(matched_files.get('existing_bbappend', {}).keys())}

Provide 3-5 specific fixes. Each must:
- Start with •
- Include exact file paths and complete commands/content
- End with (+X% confidence)
- Be universally applicable to any Yocto project

Generate complete, actionable solutions:"""

        print("🤖 FALLBACK: Using simplified AI prompt...")
        ai_response = llm.invoke(fallback_prompt)
        
        suggestions = parse_universal_ai_response(ai_response)
        
        if suggestions:
            print(f"🤖 FALLBACK: Generated {len(suggestions)} AI-driven suggestions")
            return suggestions[:5]
        
    except Exception as e:
        print(f"❌ FALLBACK: AI fallback failed: {e}")
    
    # Final emergency - return empty list (no hardcoded suggestions)
    return []


def generate_immediate_specific_suggestions(specific_details: Dict, matched_files: Dict) -> List[str]:
    """
    FIXED: Generate immediate specific suggestions without AI
    """
    suggestions = []
    
    # Handle existing .bbappend files
    for bbappend_file, file_path in matched_files['existing_bbappend'].items():
        suggestions.append(f"Fix {bbappend_file}: Edit {file_path} - Check FILESEXTRAPATHS_prepend syntax, ensure ':=' operator is used, verify file structure (+18% confidence)")
    
    # Handle missing .bbappend files
    for bbappend_file in matched_files['missing_bbappend']:
        base_name = bbappend_file.replace('.bbappend', '').split('_')[0]
        suggestions.append(f"Create missing {bbappend_file}: Create file meta-myhello/recipes-*/{base_name}/{bbappend_file} with FILESEXTRAPATHS_prepend := \"${{THISDIR}}/${{PN}}:\" (+20% confidence)")
    
    # Handle environment issues
    if specific_details['environment_issues']:
        suggestions.append("Fix BitBake environment: Add 'source oe-init-build-env build' command before BitBake execution in Jenkins pipeline (+22% confidence)")
    
    # Handle disk issues
    if specific_details['disk_issues']:
        suggestions.append("Fix disk space: Clean build artifacts with 'rm -rf tmp sstate-cache' or increase storage allocation (+15% confidence)")
    
    return suggestions[:5]



def analyze_workspace_fast(workspace_path: str, max_time_seconds: int = 5) -> Dict:
    """
    FAST workspace analysis with strict time limits
    """
    import time
    
    start_time = time.time()
    context = {
        'workspace_exists': False,
        'workspace_path': workspace_path,
        'discovered_files': [],
        'file_contents': {},
        'analysis_time': 0
    }
    
    if not workspace_path or not os.path.exists(workspace_path):
        return context
    
    context['workspace_exists'] = True
    files_scanned = 0
    max_files = 50  # STRICT LIMIT
    
    try:
        for root, dirs, files in os.walk(workspace_path):
            # Check timeout
            if time.time() - start_time > max_time_seconds:
                print(f" Workspace analysis timeout after {max_time_seconds}s")
                break
            
            # Limit depth
            depth = root.replace(workspace_path, '').count(os.sep)
            if depth > 3:
                continue
            
            # Skip large directories
            dirs[:] = [d for d in dirs if d not in ['tmp', 'sstate-cache', 'downloads', 'cache', '.git']]
            
            for file_name in files:
                if files_scanned >= max_files:
                    break
                
                file_path = os.path.join(root, file_name)
                
                # Only process small, relevant files
                if should_process_file_fast(file_name, file_path):
                    context['discovered_files'].append({
                        'name': file_name,
                        'relative_path': os.path.relpath(file_path, workspace_path)
                    })
                    
                    # Read content of critical files only
                    if file_name.endswith(('.bbappend', '.conf')) and len(context['file_contents']) < 5:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(2000)  # Max 2KB per file
                                context['file_contents'][os.path.relpath(file_path, workspace_path)] = content
                        except:
                            pass
                
                files_scanned += 1
    
    except Exception as e:
        print(f" Fast workspace analysis error: {e}")
    
    context['analysis_time'] = time.time() - start_time
    print(f" Fast workspace analysis: {files_scanned} files in {context['analysis_time']:.2f}s")
    
    return context

def should_process_file_fast(file_name: str, file_path: str) -> bool:
    """
    FAST file filtering - only process critical files
    """
    # File size check (skip large files)
    try:
        if os.path.getsize(file_path) > 100000:  # Skip files > 100KB
            return False
    except:
        return False
    
    # Only process critical Yocto files
    critical_extensions = {'.bbappend', '.conf', '.bb'}
    critical_names = {'local.conf', 'bblayers.conf', 'layer.conf'}
    
    return (any(file_name.endswith(ext) for ext in critical_extensions) or 
            file_name in critical_names)



def extract_specific_files_from_rules(violated_rules: List[Dict]) -> List[str]:
    """
    FAST: Extract specific file names from failed rules
    """
    specific_files = []
    
    for rule in violated_rules:
        rule_text = rule.get('rule_text', '')
        
        # Extract .bbappend files
        bbappend_matches = re.findall(r'([a-zA-Z0-9_-]+(?:_[\d\.]+)?\.bbappend)', rule_text)
        specific_files.extend(bbappend_matches)
        
        # Extract other specific files
        if 'local.conf' in rule_text.lower():
            specific_files.append('local.conf')
        if 'bblayers.conf' in rule_text.lower():
            specific_files.append('bblayers.conf')
        if 'layer.conf' in rule_text.lower():
            specific_files.append('layer.conf')
    
    return list(set(specific_files))  # Remove duplicates

def find_specific_files_in_workspace(specific_files: List[str], workspace_context: Dict) -> Dict:
    """
    FAST: Find specific files in workspace context
    """
    found_files = {}
    
    # Check discovered files
    for file_info in workspace_context.get('discovered_files', []):
        file_name = file_info['name']
        
        for specific_file in specific_files:
            if specific_file in file_name or file_name.endswith(specific_file):
                found_files[specific_file] = {
                    'actual_name': file_name,
                    'relative_path': file_info['relative_path'],
                    'detected_issues': []
                }
                
                # Add content if available
                rel_path = file_info['relative_path']
                if rel_path in workspace_context.get('file_contents', {}):
                    content = workspace_context['file_contents'][rel_path]
                    found_files[specific_file]['content_preview'] = content
                    
                    # Quick issue detection
                    if file_name.endswith('.bbappend'):
                        if 'FILESEXTRAPATHS' not in content:
                            found_files[specific_file]['detected_issues'].append('Missing FILESEXTRAPATHS_prepend')
                        if ':=' not in content and 'FILESEXTRAPATHS' in content:
                            found_files[specific_file]['detected_issues'].append('Possible assignment operator issue')
    
    return found_files

def parse_enhanced_suggestions(ai_response: str) -> List[str]:
    """
    FAST: Parse AI response with enhanced patterns
    """
    suggestions = []
    
    lines = ai_response.split('\n')
    for line in lines:
        line = line.strip()
        
        # Look for bullet points with confidence
        if line.startswith('•') and '(+' in line and '%' in line:
            suggestion = line[1:].strip()
            if len(suggestion) > 40:  # Ensure meaningful content
                suggestions.append(suggestion)
        
        # Also look for lines with file paths and actions
        elif any(action in line.lower() for action in ['edit', 'fix', 'add', 'create']) and \
             any(file_ext in line.lower() for file_ext in ['.bbappend', '.conf']) and \
             '(+' in line:
            suggestions.append(line.strip())
    
    return suggestions

def generate_smart_fallback_suggestions(violated_rules: List[Dict], workspace_context: Dict) -> List[str]:
    """
    SMART: Generate fallback suggestions using available context
    """
    suggestions = []
    
    # Extract file names from workspace context
    available_files = list(workspace_context.get('file_contents', {}).keys())
    
    for rule in violated_rules[:3]:
        rule_text = rule.get('rule_text', '').lower()
        
        if 'bbappend' in rule_text:
            # Find .bbappend files in workspace
            bbappend_files = [f for f in available_files if f.endswith('.bbappend')]
            
            if bbappend_files:
                for bbappend_file in bbappend_files[:2]:  # Max 2 files
                    suggestions.append(f"Fix {os.path.basename(bbappend_file)}: Edit {bbappend_file} - Check FILESEXTRAPATHS_prepend syntax and ensure proper variable assignments (+15% confidence)")
            else:
                recipe_match = re.search(r'([a-zA-Z0-9_-]+)\.bbappend', rule.get('rule_text', ''))
                if recipe_match:
                    suggestions.append(f"Create missing {recipe_match.group(1)}.bbappend: Add file to meta-*/recipes-*/ with FILESEXTRAPATHS_prepend declaration (+20% confidence)")
        
        elif 'environment' in rule_text:
            suggestions.append("Fix BitBake environment: Add 'source oe-init-build-env' before BitBake commands (+20% confidence)")
        
        elif 'machine' in rule_text:
            local_conf_files = [f for f in available_files if 'local.conf' in f]
            if local_conf_files:
                suggestions.append(f"Fix machine config: Edit {local_conf_files[0]} - Add MACHINE = \"qemux86-64\" (+18% confidence)")
            else:
                suggestions.append("Create local.conf: Add file with MACHINE = \"qemux86-64\" (+18% confidence)")
    
    return suggestions[:5]




def detect_workspace_patterns_for_ai(context: Dict) -> List[str]:
    """
    Detect workspace patterns to help AI understand the build system
    """
    patterns = []
    
    discovered_files = context.get('discovered_files', [])
    file_names = [f['name'] for f in discovered_files]
    file_extensions = [f['extension'] for f in discovered_files]
    
    # Count file types
    extension_counts = {}
    for ext in file_extensions:
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    patterns.append(f"File types distribution: {extension_counts}")
    
    # Detect directory patterns
    directories = list(context.get('directory_structure', {}).keys())
    patterns.append(f"Directory structure: {directories[:20]}")  # Limit for readability
    
    # Detect naming patterns
    bb_files = [f for f in file_names if f.endswith('.bb')]
    if bb_files:
        patterns.append(f"BitBake recipes found: {len(bb_files)} files")
    
    bbappend_files = [f for f in file_names if f.endswith('.bbappend')]
    if bbappend_files:
        patterns.append(f"Recipe modifications found: {len(bbappend_files)} .bbappend files")
    
    conf_files = [f for f in file_names if f.endswith('.conf')]
    if conf_files:
        patterns.append(f"Configuration files found: {conf_files}")
    
    return patterns




def build_ai_context_summary(violated_rules: List[Dict], pipeline_text: str, workspace_context: Dict) -> str:
    """
    Build comprehensive context summary for AI with maximum detail
    """
    summary_parts = []
    
    # Workspace overview
    if workspace_context.get('workspace_exists'):
        summary_parts.append(f"WORKSPACE: {workspace_context.get('workspace_path', 'Unknown')}")
        summary_parts.append(f"TOTAL FILES: {len(workspace_context.get('discovered_files', []))}")
        
        # File type breakdown
        discovered_files = workspace_context.get('discovered_files', [])
        file_types = {}
        for file_info in discovered_files:
            ext = file_info.get('extension', 'no-ext')
            file_types[ext] = file_types.get(ext, 0) + 1
        
        summary_parts.append(f"FILE TYPES: {file_types}")
        
        # Directory structure
        dirs = list(workspace_context.get('directory_structure', {}).keys())[:10]
        summary_parts.append(f"DIRECTORIES: {dirs}")
        
        # Specific Yocto files
        yocto_files = [f['relative_path'] for f in discovered_files 
                      if f['name'].endswith(('.bb', '.bbappend', '.conf'))]
        if yocto_files:
            summary_parts.append(f"YOCTO FILES: {yocto_files}")
    else:
        summary_parts.append("WORKSPACE: Not accessible")
    
    # Available file contents
    file_contents = workspace_context.get('file_contents', {})
    if file_contents:
        summary_parts.append(f"READABLE FILES: {list(file_contents.keys())}")
    
    return "\n".join(summary_parts)








def deduplicate_rules(rules: List[Dict]) -> List[Dict]:
    """Remove duplicate rules based on content similarity"""
    unique_rules = {}
    
    for rule in rules:
        rule_text = rule['rule_text'].strip()
        
        # Create a key based on first 50 characters (normalized)
        key = re.sub(r'[^\w\s]', '', rule_text.lower())[:50].strip()
        
        if key and key not in unique_rules:
            unique_rules[key] = rule
        elif key in unique_rules and rule.get('confidence', 0.5) > unique_rules[key].get('confidence', 0.5):
            # Keep the higher confidence version
            unique_rules[key] = rule
    
    return list(unique_rules.values())

def filter_quality_rules(rules: List[Dict]) -> List[Dict]:
    """Filter out low-quality and meaningless rule fragments"""
    quality_rules = []
    
    for rule in rules:
        rule_text = rule['rule_text'].strip()
        
        # Skip very short or meaningless rules
        if len(rule_text) < 20:
            continue
            
        # Skip fragment rules
        if any(fragment in rule_text.lower() for fragment in [
            '**location**:', '**contents**:', 'pipeline] {', '+ echo', '...'
        ]):
            continue
            
        # Skip rules that are mostly punctuation
        alpha_chars = sum(1 for c in rule_text if c.isalpha())
        if alpha_chars < len(rule_text) * 0.6:  # Less than 60% alphabetic
            continue
            
        # Only keep rules with meaningful content
        if any(meaningful_word in rule_text.lower() for meaningful_word in [
            'must', 'should', 'ensure', 'configure', 'build', 'install', 'deploy', 
            'environment', 'permission', 'timeout', 'storage', 'directory', 'agent',
            'repository', 'branch', 'image', 'container', 'service', 'stage', 'step'
        ]):
            quality_rules.append(rule)
    
    return quality_rules

def evaluate_rule_against_pipeline_adaptive(rule: Dict, pipeline_text: str, pipeline_lower: str) -> bool:
    """
    FULLY ADAPTIVE rule evaluation - learns from knowledge base content
    NO HARDCODING - works for ANY project type with ANY knowledge base
    """
    rule_text = rule['rule_text']
    rule_text_lower = rule_text.lower()
    
    # EXTRACT REQUIREMENTS AND EXPECTATIONS FROM THE RULE ITSELF
    requirements = extract_requirements_from_rule(rule_text)
    expectations = extract_expectations_from_rule(rule_text)
    constraints = extract_constraints_from_rule(rule_text)
    
    # EVALUATE EACH REQUIREMENT AGAINST PIPELINE
    satisfied_requirements = 0
    total_requirements = len(requirements) + len(expectations) + len(constraints)
    
    if total_requirements == 0:
        # Fall back to semantic similarity if no specific requirements found
        return evaluate_semantic_similarity(rule_text_lower, pipeline_lower)
    
    # Check requirements
    for requirement in requirements:
        if evaluate_requirement_presence(requirement, pipeline_lower):
            satisfied_requirements += 1
        else:
            print(f"  ⚠️ Requirement not met: {requirement}")
    
    # Check expectations  
    for expectation in expectations:
        if evaluate_expectation_fulfillment(expectation, pipeline_text, pipeline_lower):
            satisfied_requirements += 1
        else:
            print(f"  ⚠️ Expectation not met: {expectation}")
    
    # Check constraints
    for constraint in constraints:
        if evaluate_constraint_compliance(constraint, pipeline_text, pipeline_lower):
            satisfied_requirements += 1
        else:
            print(f"  ⚠️ Constraint violated: {constraint}")
    
    # Rule is satisfied if >= 70% of requirements are met
    satisfaction_ratio = satisfied_requirements / total_requirements
    return satisfaction_ratio >= 0.7

def extract_requirements_from_rule(rule_text: str) -> List[str]:
    """Extract explicit requirements from rule text"""
    import re
    
    requirements = []
    
    # Pattern: "must", "should", "requires", "needs"
    requirement_patterns = [
        r'must\s+([^.!?]+)',
        r'should\s+([^.!?]+)', 
        r'requires?\s+([^.!?]+)',
        r'needs?\s+([^.!?]+)',
        r'ensure\s+([^.!?]+)',
        r'verify\s+([^.!?]+)'
    ]
    
    for pattern in requirement_patterns:
        matches = re.findall(pattern, rule_text, re.IGNORECASE)
        requirements.extend([match.strip() for match in matches])
    
    return requirements

def extract_expectations_from_rule(rule_text: str) -> List[str]:
    """Extract expectations and best practices from rule text"""
    import re
    
    expectations = []
    
    # Pattern: "recommended", "suggested", "advised", "preferred"
    expectation_patterns = [
        r'recommended\s+([^.!?]+)',
        r'suggested\s+([^.!?]+)',
        r'advised\s+([^.!?]+)', 
        r'preferred\s+([^.!?]+)',
        r'consider\s+([^.!?]+)',
        r'best\s+practice\s+([^.!?]+)'
    ]
    
    for pattern in expectation_patterns:
        matches = re.findall(pattern, rule_text, re.IGNORECASE)
        expectations.extend([match.strip() for match in matches])
    
    return expectations

def extract_constraints_from_rule(rule_text: str) -> List[str]:
    """Extract constraints and limits from rule text"""
    import re
    
    constraints = []
    
    # Pattern: "at least", "minimum", "maximum", "no more than", numeric values
    constraint_patterns = [
        r'at\s+least\s+([^.!?]+)',
        r'minimum\s+([^.!?]+)',
        r'maximum\s+([^.!?]+)', 
        r'no\s+more\s+than\s+([^.!?]+)',
        r'(\d+)\s*(gb|mb|hours?|minutes?|threads?)',
        r'avoid\s+([^.!?]+)',
        r'never\s+([^.!?]+)'
    ]
    
    for pattern in constraint_patterns:
        matches = re.findall(pattern, rule_text, re.IGNORECASE)
        if isinstance(matches[0] if matches else None, tuple):
            # Handle tuples from numeric patterns
            constraints.extend([f"{match[0]} {match[1]}" for match in matches])
        else:
            constraints.extend([match.strip() for match in matches])
    
    return constraints

def evaluate_requirement_presence(requirement: str, pipeline_lower: str) -> bool:
    """Check if a requirement is present in pipeline"""
    requirement_lower = requirement.lower()
    
    # Extract key terms from requirement
    key_terms = extract_key_technical_terms(requirement_lower)
    
    if not key_terms:
        # Fall back to basic substring matching
        return any(word in pipeline_lower for word in requirement_lower.split() if len(word) > 3)
    
    # Require at least 60% of key terms to be present
    matches = sum(1 for term in key_terms if term in pipeline_lower)
    return matches / len(key_terms) >= 0.6

def evaluate_expectation_fulfillment(expectation: str, pipeline_text: str, pipeline_lower: str) -> bool:
    """Check if an expectation is fulfilled"""
    expectation_lower = expectation.lower()
    
    # Extract action words and concepts
    actions = extract_action_words(expectation_lower)
    concepts = extract_technical_concepts(expectation_lower)
    
    # More lenient matching for expectations (50%)
    all_terms = actions + concepts
    if not all_terms:
        return True  # Can't evaluate, assume fulfilled
    
    matches = sum(1 for term in all_terms if term in pipeline_lower)
    return matches / len(all_terms) >= 0.5

def evaluate_constraint_compliance(constraint: str, pipeline_text: str, pipeline_lower: str) -> bool:
    """Check if a constraint is complied with"""
    constraint_lower = constraint.lower()
    
    # Handle numeric constraints
    if any(char.isdigit() for char in constraint):
        return evaluate_numeric_constraint(constraint_lower, pipeline_text)
    
    # Handle avoidance constraints
    if any(avoid_word in constraint_lower for avoid_word in ['avoid', 'never', 'don\'t', 'not']):
        return evaluate_avoidance_constraint(constraint_lower, pipeline_lower)
    
    # Default semantic matching
    key_terms = extract_key_technical_terms(constraint_lower)
    if not key_terms:
        return True
    
    matches = sum(1 for term in key_terms if term in pipeline_lower)
    return matches / len(key_terms) >= 0.6

def evaluate_numeric_constraint(constraint: str, pipeline_text: str) -> bool:
    """Evaluate numeric constraints dynamically"""
    import re
    
    # Extract numbers and units from constraint
    constraint_numbers = re.findall(r'(\d+)\s*([a-z]+)?', constraint.lower())
    pipeline_numbers = re.findall(r'(\d+)\s*([a-z]+)?', pipeline_text.lower())
    
    if not constraint_numbers or not pipeline_numbers:
        return True  # Can't evaluate
    
    # Simple heuristic: if pipeline has larger numbers, likely satisfies constraint
    constraint_max = max(int(num[0]) for num in constraint_numbers)
    pipeline_max = max(int(num[0]) for num in pipeline_numbers)
    
    return pipeline_max >= constraint_max * 0.8  # 80% threshold

def evaluate_avoidance_constraint(constraint: str, pipeline_lower: str) -> bool:
    """Check if things to avoid are actually avoided"""
    
    # Extract what should be avoided
    avoid_terms = []
    constraint_words = constraint.split()
    
    for i, word in enumerate(constraint_words):
        if word in ['avoid', 'never', 'don\'t', 'not']:
            # Get the next few words as things to avoid
            avoid_terms.extend(constraint_words[i+1:i+4])
    
    # Check if avoided things are present (bad)
    for term in avoid_terms:
        if len(term) > 3 and term in pipeline_lower:
            return False  # Constraint violated
    
    return True  # Constraint satisfied

def extract_key_technical_terms(text: str) -> List[str]:
    """Extract technical terms from text"""
    import re
    
    # Technical patterns
    patterns = [
        r'\b[a-z]+[-_][a-z]+\b',      # hyphenated terms
        r'\b[A-Z]{2,}\b',             # acronyms
        r'\b\w*\d+\w*\b',             # terms with numbers
        r'\b(?:config|setup|build|deploy|install|create|manage|run|execute|start|stop|enable|disable|configure|prepare|initialize|validate|verify|check|ensure|monitor|log|debug|test|compile|package|archive|publish|release|version|branch|commit|merge|clone|fetch|push|pull)\b'
    ]
    
    terms = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        terms.extend(matches)
    
    return list(set(terms))

def extract_action_words(text: str) -> List[str]:
    """Extract action/verb words from text"""
    action_words = [
        'build', 'deploy', 'install', 'configure', 'setup', 'create', 'manage',
        'run', 'execute', 'start', 'stop', 'enable', 'disable', 'prepare', 
        'initialize', 'validate', 'verify', 'check', 'ensure', 'monitor',
        'log', 'debug', 'test', 'compile', 'package', 'archive', 'publish'
    ]
    
    return [word for word in action_words if word in text]

def extract_technical_concepts(text: str) -> List[str]:
    """Extract technical concepts from text"""
    import re
    
    # Extract quoted terms, paths, and technical identifiers
    concepts = []
    
    # Quoted terms
    quoted = re.findall(r'[\'"]([^\'"]+)[\'"]', text)
    concepts.extend(quoted)
    
    # File paths
    paths = re.findall(r'/[\w\-/\.]+', text)
    concepts.extend(paths)
    
    # Technical identifiers (camelCase, snake_case, etc.)
    identifiers = re.findall(r'\b[a-z]+[_-][a-z]+\b|\b[a-z]+[A-Z][a-z]+\b', text)
    concepts.extend(identifiers)
    
    return [concept.lower() for concept in concepts if len(concept) > 2]

def evaluate_semantic_similarity(rule_text: str, pipeline_text: str) -> bool:
    """Fallback semantic similarity evaluation"""
    rule_terms = set(extract_key_technical_terms(rule_text))
    pipeline_terms = set(extract_key_technical_terms(pipeline_text))
    
    if not rule_terms:
        return True
    
    overlap = len(rule_terms.intersection(pipeline_terms))
    similarity = overlap / len(rule_terms)
    
    return similarity >= 0.4  # 40% similarity threshold

def generate_adaptive_response(
    analysis: Dict, technologies: List[str], rules: List[Dict], 
    success_patterns: List[str], failure_patterns: List[str]
) -> str:
    """Generate response using fully adaptive analysis"""
    
    total_rules = len([r for r in analysis['evaluated_rules']])
    violation_ratio = analysis['violated_count'] / total_rules if total_rules > 0 else 0
    
    # ADAPTIVE PREDICTION based on violation ratio and pattern analysis
    if violation_ratio >= 0.5:
        prediction = "FAIL"
        confidence = max(25, int(60 * (1 - violation_ratio)))
    elif violation_ratio >= 0.3:
        prediction = "HIGH_RISK"  
        confidence = max(50, int(75 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(70, int(90 * (1 - violation_ratio)))
    
    # Adjust based on historical patterns
    if len(failure_patterns) > len(success_patterns):
        prediction = "HIGH_RISK" if prediction == "PASS" else prediction
        confidence = max(confidence - 10, 25)
    
    # Build response
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES:
"""
    
    for rule_text in analysis['evaluated_rules'][:25]:
        response += f"{rule_text}\n"
    
    response += f"""HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
{success_patterns[0] if success_patterns else "✅ No specific success patterns found"}
{success_patterns[1] if len(success_patterns) > 1 else "✅ Standard configuration detected"}
FAILURE PATTERNS:
{failure_patterns[0] if failure_patterns else "❌ No specific failure patterns found"}  
{failure_patterns[1] if len(failure_patterns) > 1 else "❌ Configuration issues possible"}
APPLICABLE_RULES: {total_rules}
SATISFIED_RULES: {analysis['satisfied_count']}
VIOLATED_RULES: {analysis['violated_count']}
RISK_FACTORS: {'; '.join(analysis['critical_violations'][:3]) if analysis['critical_violations'] else 'None detected'}
PREDICTION: {prediction}
CONFIDENCE: {confidence}%
REASONING: Adaptive knowledge-driven analysis using {total_rules} rules extracted from vectorstore. Analysis based on requirement patterns, constraints, and expectations found in knowledge base."""
    
    return response


def extract_meaningful_terms_improved(rule_text: str) -> List[str]:
    """Improved meaningful term extraction"""
    
    # Extract important technical terms
    important_patterns = [
        r'\b(?:agent|label|stage|build|test|deploy|environment|timeout|permission|chown|chmod|bitbake|yocto|jenkins|docker|git|branch|repository|image|container|service|port|directory|file|configuration|install|setup|prepare|configure|compile|package|archive|publish|validate|verify)\b'
    ]
    
    terms = []
    for pattern in important_patterns:
        matches = re.findall(pattern, rule_text.lower())
        terms.extend(matches)
    
    # Also extract file/path patterns
    file_patterns = re.findall(r'\b\w+\.\w{2,4}\b', rule_text)  # file.ext patterns
    path_patterns = re.findall(r'/[\w\-/]+', rule_text)         # /path/to/something patterns
    
    terms.extend([f.lower() for f in file_patterns])
    terms.extend([p.lower() for p in path_patterns])
    
    # Remove duplicates and return
    return list(set(terms))[:6]  # Top 6 terms

def generate_improved_response(
    analysis: Dict, technologies: List[str], rules: List[Dict], 
    success_patterns: List[str], failure_patterns: List[str]
) -> str:
    """Generate improved response with better prediction logic"""
    
    total_rules = len([r for r in analysis['evaluated_rules']])  # Count actual evaluated rules
    violation_ratio = analysis['violated_count'] / total_rules if total_rules > 0 else 0
    
    # IMPROVED PREDICTION LOGIC
    if len(analysis['critical_violations']) >= 3 or violation_ratio >= 0.6:
        prediction = "FAIL"
        confidence = max(25, int(70 * (1 - violation_ratio)))
    elif len(analysis['critical_violations']) >= 1 or violation_ratio >= 0.4:
        prediction = "HIGH_RISK"  
        confidence = max(50, int(80 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(75, int(95 * (1 - violation_ratio)))
    
    # Build response
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES:
"""
    
    # Show rules without duplication
    for rule_text in analysis['evaluated_rules'][:25]:  # Limit to 25 rules for readability
        response += f"{rule_text}\n"
    
    response += f"""HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
{success_patterns[0] if success_patterns else "✅ No specific success patterns found"}
{success_patterns[1] if len(success_patterns) > 1 else "✅ Standard configuration detected"}
FAILURE PATTERNS:
{failure_patterns[0] if failure_patterns else "❌ No specific failure patterns found"}  
{failure_patterns[1] if len(failure_patterns) > 1 else "❌ Configuration issues possible"}
APPLICABLE_RULES: {total_rules}
SATISFIED_RULES: {analysis['satisfied_count']}
VIOLATED_RULES: {analysis['violated_count']}
RISK_FACTORS: {'; '.join(analysis['critical_violations'][:3]) if analysis['critical_violations'] else 'None detected'}
PREDICTION: {prediction}
CONFIDENCE: {confidence}%
REASONING: Enhanced adaptive analysis using {total_rules} quality rules. {len(analysis['present_stages'])} stages detected. Pipeline appears properly configured for Yocto build."""
    
    return response

def extract_stages_from_vectorstore_rules(rules: List[Dict]) -> List[str]:
    """Learn what stages should exist from vectorstore documents (MinIO content)"""
    stages = []
    
    for rule in rules:
        rule_text_lower = rule['rule_text'].lower()
        
        # Extract stages mentioned in vectorstore documents
        import re
        stage_terms = re.findall(r'stage[:\s]*[\'"]([^\'"]+)[\'"]', rule_text_lower)
        stages.extend([stage.strip() for stage in stage_terms])
        
        # Look for process words that indicate stages
        process_indicators = ['clone', 'build', 'test', 'deploy', 'setup', 'configure', 'prepare', 'package', 'archive', 'publish']
        for indicator in process_indicators:
            if indicator in rule_text_lower and ('stage' in rule_text_lower or 'step' in rule_text_lower):
                stages.append(indicator)
    
    return list(set(stages))  # Remove duplicates

# CORRECTED MAIN FUNCTION
# def sync_get_rag_response(query: str) -> str:
#     print("🧠 KNOWLEDGE-BASE DRIVEN ANALYSIS")
#     print("📚 Knowledge Base = Vectorstore (populated from MinIO)")
    
#     # Check if this is a pipeline analysis request
#     is_pipeline_query = any(keyword in query.lower() for keyword in [
#         'pipeline', 'stage', 'steps', 'bitbake', 'yocto', 'jenkins', 'build', 'workflow'
#     ])
    
#     if not is_pipeline_query:
#         return "No pipeline detected - please provide a pipeline to analyze"
    
#     # STEP 1: Retrieve relevant documents from vectorstore (knowledge base)
#     all_docs = vectorstore.similarity_search(query, k=20)
#     print(f"📚 Retrieved {len(all_docs)} documents from vectorstore (MinIO knowledge base)")
    
#     # STEP 2: Extract historical patterns from LOG documents in vectorstore
#     try:
#         success_patterns, failure_patterns = get_historical_patterns_from_logs()
#     except Exception as e:
#         print(f"❌ Error extracting log patterns from vectorstore: {e}")
#         success_patterns = ["✅ No historical success patterns found in logs"]
#         failure_patterns = ["❌ No historical failure patterns found in logs"]
    
#     # STEP 3: Extract intelligent rules from ALL vectorstore documents
#     all_intelligent_rules = []
#     print("🔧 Extracting rules from vectorstore documents (MinIO content)...")
    
#     for doc in all_docs:
#         content = doc.page_content
#         doc_source = doc.metadata.get('source', 'unknown_minio_file')
#         doc_category = doc.metadata.get('category', 'documentation')
        
#         # Skip our own predictions to avoid loops
#         if any(skip in content for skip in ["DETECTED_STACK:", "PREDICTION:", "ESTABLISHED_RULES:"]):
#             continue
        
#         try:
#             # Extract rules from this vectorstore document
#             intelligent_rules = extract_lightweight_intelligent_rules(content, doc_category)
            
#             # Add source information from document metadata
#             for rule in intelligent_rules:
#                 rule['source'] = doc_source
#                 rule['category'] = doc_category
            
#             all_intelligent_rules.extend(intelligent_rules)
#             print(f"  📋 Extracted {len(intelligent_rules)} rules from {doc_source}")
            
#         except Exception as e:
#             print(f"  ❌ Error extracting rules from {doc_source}: {e}")
#             continue
    
#     print(f"🎯 Total rules extracted from vectorstore: {len(all_intelligent_rules)}")
    
#     # STEP 4: ADAPTIVE ANALYSIS - Learn from vectorstore content
#     pipeline_analysis = analyze_pipeline_with_adaptive_intelligence(query, all_intelligent_rules)
    
#     # STEP 5: Detect technology stack from pipeline
#     detected_tech = detect_technology_stack_enhanced(query)
    
#     # STEP 6: Generate response using vectorstore-driven analysis
#     return generate_adaptive_response(
#         pipeline_analysis, detected_tech, all_intelligent_rules, 
#         success_patterns, failure_patterns
#     )


def evaluate_rule_against_pipeline(rule: Dict, pipeline_text: str, pipeline_lower: str) -> bool:
    """
    Intelligently evaluate if a rule is satisfied by the pipeline
    NO HARDCODING - Pure pattern matching and semantic analysis
    """
    rule_text_lower = rule['rule_text'].lower()
    rule_type = rule.get('rule_type', 'UNKNOWN')
    
    # SEMANTIC ANALYSIS - Extract key concepts from rule text
    key_concepts = extract_key_concepts_from_rule(rule_text_lower)
    
    # REQUIREMENT DETECTION - Look for requirement patterns
    if rule_type in ['MANDATORY', 'REQUIREMENT', 'VERIFICATION']:
        # Check if pipeline contains the required elements
        return any(concept in pipeline_lower for concept in key_concepts)
    
    elif rule_type in ['CONFIGURATION', 'ENVIRONMENT_VAR']:
        # Check for configuration presence
        config_terms = extract_configuration_terms(rule_text_lower)
        return any(term in pipeline_lower for term in config_terms)
    
    elif rule_type in ['STORAGE_REQUIREMENT', 'CONSTRAINT']:
        # Check for resource/constraint satisfaction
        constraint_terms = extract_constraint_terms(rule_text_lower)
        return any(term in pipeline_lower for term in constraint_terms)
    
    elif rule_type in ['NUMBERED_STEP', 'PROCESS', 'PROCEDURAL']:
        # Check for process step presence
        process_terms = extract_process_terms(rule_text_lower)
        return any(term in pipeline_lower for term in process_terms)
    
    else:
        # GENERIC SEMANTIC MATCHING - Extract meaningful terms
        meaningful_terms = extract_meaningful_terms(rule_text_lower)
        if not meaningful_terms:
            return True  # If we can't extract terms, assume satisfied
            
        # Require at least 50% of meaningful terms to be present
        matches = sum(1 for term in meaningful_terms if term in pipeline_lower)
        match_ratio = matches / len(meaningful_terms)
        return match_ratio >= 0.5

def extract_key_concepts_from_rule(rule_text: str) -> List[str]:
    """Extract key concepts from rule text using NLP-like approach"""
    
    # Remove common stop words and extract meaningful terms
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'is', 'are', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
        'did', 'will', 'would', 'could', 'should', 'must', 'shall', 'may'
    }
    
    # Extract words (3+ characters)
    words = re.findall(r'\b\w{3,}\b', rule_text.lower())
    
    # Filter out stopwords and extract meaningful terms
    concepts = []
    for word in words:
        if word not in stopwords:
            concepts.append(word)
    
    return concepts[:5]  # Top 5 most important concepts

def extract_configuration_terms(rule_text: str) -> List[str]:
    """Extract configuration-related terms"""
    config_patterns = [
        r'[A-Z_]+(?:\s*=|\s*:)',  # Environment variables
        r'\w+\.conf\b',           # Configuration files
        r'\w+\.properties\b',     # Properties files
        r'\w+\.yaml\b|\w+\.yml\b', # YAML files
        r'port\s*\d+',            # Port numbers
        r'timeout\s*\d+',         # Timeout values
    ]
    
    terms = []
    for pattern in config_patterns:
        matches = re.findall(pattern, rule_text, re.IGNORECASE)
        terms.extend(matches)
    
    return terms

def extract_constraint_terms(rule_text: str) -> List[str]:
    """Extract constraint/resource terms"""
    constraint_patterns = [
        r'\d+\s*(?:gb|mb|tb)',     # Storage amounts
        r'\d+\s*(?:hours?|minutes?)', # Time amounts
        r'\d+\s*(?:cores?|threads?)', # CPU amounts
        r'minimum\s+\w+',          # Minimum requirements
        r'maximum\s+\w+',          # Maximum limits
        r'at\s+least\s+\w+',       # At least requirements
    ]
    
    terms = []
    for pattern in constraint_patterns:
        matches = re.findall(pattern, rule_text, re.IGNORECASE)
        terms.extend(matches)
    
    return [term.split()[-1] for term in terms if term]  # Extract the key term

def extract_process_terms(rule_text: str) -> List[str]:
    """Extract process/action terms"""
    action_words = [
        'clone', 'build', 'test', 'deploy', 'install', 'configure', 'setup',
        'prepare', 'compile', 'package', 'archive', 'publish', 'validate',
        'verify', 'check', 'ensure', 'execute', 'run', 'start', 'stop'
    ]
    
    terms = []
    for word in action_words:
        if word in rule_text:
            terms.append(word)
    
    return terms

def extract_meaningful_terms(rule_text: str) -> List[str]:
    """Extract meaningful terms for generic matching"""
    
    # Technical terms are usually longer and contain specific patterns
    technical_patterns = [
        r'\b[a-z]+[-_][a-z]+\b',    # hyphenated or underscored terms
        r'\b[A-Z]{2,}\b',           # All caps acronyms
        r'\b\w*[0-9]\w*\b',         # Terms with numbers
        r'\b\w{6,}\b',              # Longer technical terms
    ]
    
    meaningful_terms = []
    for pattern in technical_patterns:
        matches = re.findall(pattern, rule_text)
        meaningful_terms.extend([match.lower() for match in matches])
    
    # Also include important domain-specific terms
    domain_terms = re.findall(r'\b(?:pipeline|stage|build|test|deploy|agent|timeout|environment|config|setup|install|docker|jenkins|kubernetes|git|branch|repository|image|container|service|port|path|directory|file|script|command|tool|framework|platform|system|server|client|api|endpoint|database|cache|storage|memory|cpu|disk|network|security|authentication|authorization|monitoring|logging|backup|restore|scale|performance|optimization|automation|integration|deployment|release|version|branch|commit|merge|pull|push|clone|checkout|fetch|diff|status|log|tag|stash)\b', rule_text.lower())
    
    meaningful_terms.extend(domain_terms)
    
    # Remove duplicates and return top terms
    return list(set(meaningful_terms))[:8]

def extract_stages_from_vectorstore_rules(rules: List[Dict]) -> List[str]:
    """Learn what stages should exist from vectorstore documents (MinIO content)"""
    stages = []
    
    for rule in rules:
        rule_text_lower = rule['rule_text'].lower()
        
        # Extract stages mentioned in vectorstore documents
        stage_terms = re.findall(r'stage[:\s]*[\'"]([^\'"]+)[\'"]', rule_text_lower)
        stages.extend([stage.strip() for stage in stage_terms])
        
        # Look for process words that indicate stages
        process_indicators = ['clone', 'build', 'test', 'deploy', 'setup', 'configure', 'prepare', 'package', 'archive', 'publish']
        for indicator in process_indicators:
            if indicator in rule_text_lower and ('stage' in rule_text_lower or 'step' in rule_text_lower):
                stages.append(indicator)
    
    return list(set(stages))  # Remove duplicates

def extract_stages_from_pipeline(pipeline_text: str) -> List[str]:
    """Extract stages actually present in the pipeline"""
    
    pipeline_lower = pipeline_text.lower()
    
    # Extract Jenkins-style stages
    jenkins_stages = re.findall(r"stage\s*\(\s*['\"]([^'\"]+)['\"]", pipeline_lower)
    
    # Also look for common stage keywords
    common_stages = ['clone', 'build', 'test', 'deploy', 'setup', 'configure', 'prepare', 'package', 'archive', 'publish']
    present_stages = []
    
    for stage in common_stages:
        if stage in pipeline_lower:
            present_stages.append(stage)
    
    # Combine and deduplicate
    all_stages = jenkins_stages + present_stages
    return list(set([stage.lower().strip() for stage in all_stages]))

def generate_adaptive_intelligent_response(
    analysis: Dict, technologies: List[str], rules: List[Dict], 
    success_patterns: List[str], failure_patterns: List[str]
) -> str:
    """Generate response using adaptive analysis results"""
    
    total_rules = len(rules)
    violation_ratio = analysis['violated_count'] / total_rules if total_rules > 0 else 1
    
    # Prediction logic based on violation ratio and critical violations
    if analysis['critical_violations'] and violation_ratio >= 0.4:
        prediction = "FAIL"
        confidence = max(20, int(60 * (1 - violation_ratio)))
    elif violation_ratio >= 0.3 or analysis['missing_stages']:
        prediction = "HIGH_RISK"
        confidence = max(50, int(75 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(70, int(90 * (1 - violation_ratio)))
    
    # Build response
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES:
"""
    
    # Show all rules with their status
    for rule_text in analysis['evaluated_rules']:
        response += f"{rule_text}\n"
    
    response += f"""HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
{success_patterns[0] if success_patterns else "✅ No specific success patterns found"}
{success_patterns[1] if len(success_patterns) > 1 else "✅ Standard configuration detected"}
FAILURE PATTERNS:
{failure_patterns[0] if failure_patterns else "❌ No specific failure patterns found"}
{failure_patterns[1] if len(failure_patterns) > 1 else "❌ Configuration issues possible"}
APPLICABLE_RULES: {total_rules}
SATISFIED_RULES: {analysis['satisfied_count']}
VIOLATED_RULES: {analysis['violated_count']}
RISK_FACTORS: {'; '.join(analysis['critical_violations'] + analysis['warnings']) if (analysis['critical_violations'] or analysis['warnings']) else 'None detected'}
PREDICTION: {prediction}
CONFIDENCE: {confidence}%
REASONING: Adaptive knowledge-based analysis using {total_rules} extracted rules. {len(analysis['present_stages'])} stages detected in pipeline."""
    
    if analysis['critical_violations']:
        response += f" CRITICAL: {'; '.join(analysis['critical_violations'][:3])}"
    elif analysis['warnings']:
        response += f" WARNINGS: {len(analysis['warnings'])} potential issues detected."
    
    return response

# DEBUG FUNCTION TO SHOW VECTORSTORE CONTENT
def debug_vectorstore_knowledge_base():
    """Debug function to show what's in the vectorstore (knowledge base)"""
    try:
        # Get sample documents
        sample_docs = vectorstore.similarity_search("jenkins build pipeline", k=10)
        
        knowledge_base_summary = {
            "total_documents_sampled": len(sample_docs),
            "document_sources": [],
            "document_categories": [],
            "content_types": {}
        }
        
        for doc in sample_docs:
            source = doc.metadata.get('source', 'unknown')
            category = doc.metadata.get('category', 'unknown')
            
            knowledge_base_summary["document_sources"].append(source)
            knowledge_base_summary["document_categories"].append(category)
            
            if category not in knowledge_base_summary["content_types"]:
                knowledge_base_summary["content_types"][category] = []
            
            knowledge_base_summary["content_types"][category].append({
                "source": source,
                "content_preview": doc.page_content[:100] + "..."
            })
        
        print(" VECTORSTORE (KNOWLEDGE BASE) CONTENTS:")
        print(f"   Sample Documents: {knowledge_base_summary['total_documents_sampled']}")
        print(f"   Unique Sources: {len(set(knowledge_base_summary['document_sources']))}")
        print(f"   Categories: {set(knowledge_base_summary['document_categories'])}")
        
        for category, docs in knowledge_base_summary["content_types"].items():
            print(f"   {category.upper()}: {len(docs)} documents")
            for doc in docs[:2]:  # Show first 2 examples
                print(f"    - {doc['source']}: {doc['content_preview']}")
        
        return knowledge_base_summary
        
    except Exception as e:
        print(f"❌ Error analyzing vectorstore: {e}")
        return {"error": str(e)}


def extract_key_terms_from_rule(rule_text: str) -> List[str]:
    """Extract key terms from rule text for matching"""
    import re
    
    # Remove common words and extract meaningful terms
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'must', 'shall'}
    
    words = re.findall(r'\b\w{3,}\b', rule_text.lower())
    return [word for word in words if word not in stopwords][:5]  # Return top 5 key terms

def detect_technology_stack_enhanced(query: str) -> List[str]:
    """Enhanced technology stack detection"""
    technologies = set()
    query_lower = query.lower()
    
    # Technology patterns with confidence
    tech_patterns = {
        'jenkins': ['jenkins', 'pipeline', 'jenkinsfile', 'stage', 'steps'],
        'yocto': ['yocto', 'bitbake', 'poky', 'meta-', 'bb_', 'openembedded'],
        'docker': ['docker', 'dockerfile', 'container', 'image'],
        'kubernetes': ['kubernetes', 'kubectl', 'k8s', 'deployment', 'service'],
        'git': ['git', 'github', 'gitlab', 'clone', 'checkout'],
        'linux': ['linux', 'ubuntu', 'centos', 'debian', 'kernel'],
        'build-tools': ['make', 'cmake', 'maven', 'gradle', 'ninja'],
        'testing': ['test', 'junit', 'pytest', 'coverage']
    }
    
    for tech, patterns in tech_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            technologies.add(tech)
    
    return list(technologies) if technologies else ['general-pipeline']

def generate_enhanced_intelligent_response(
    analysis: Dict, technologies: List[str], rules: List[Dict], 
    success_patterns: List[str], failure_patterns: List[str]
) -> str:
    """Generate enhanced structured response using intelligent analysis"""
    
    total_rules = len(rules)
    violation_ratio = analysis['violated_count'] / total_rules if total_rules > 0 else 1
    
    # Enhanced prediction logic
    if analysis['critical_violations'] or violation_ratio >= 0.6:
        prediction = "FAIL"
        confidence = max(20, int(80 * (1 - violation_ratio)))
    elif analysis['missing_stages'] or violation_ratio >= 0.3:
        prediction = "HIGH_RISK"
        confidence = max(60, int(70 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(80, int(90 * (1 - violation_ratio)))
    
    # Build response
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES:
"""
    
    if analysis['evaluated_rules']:
        for rule in analysis['evaluated_rules'][:30]:  # Limit for readability
            response += f"{rule}\n"
    else:
        response += "• No rules extracted from knowledge base\n"
    
    response += f"""HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
{success_patterns[0] if success_patterns else "✅ No specific success patterns found"}
{success_patterns[1] if len(success_patterns) > 1 else "✅ Standard pipeline configuration detected"}
FAILURE PATTERNS:
{failure_patterns[0] if failure_patterns else "❌ No specific failure patterns found"}
{failure_patterns[1] if len(failure_patterns) > 1 else "❌ Configuration issues possible"}
APPLICABLE_RULES: {total_rules}
SATISFIED_RULES: {analysis['satisfied_count']}
VIOLATED_RULES: {analysis['violated_count']}
RISK_FACTORS: {'; '.join(analysis['risk_factors'] + analysis['missing_stages'] + analysis['critical_violations']) if (analysis['risk_factors'] or analysis['missing_stages'] or analysis['critical_violations']) else 'None detected'}
PREDICTION: {prediction}
CONFIDENCE: {confidence}%
REASONING: Enhanced intelligent analysis using {total_rules} platform-agnostic rules. {len(analysis['present_stages'])} out of 4 critical stages present."""
    
    if analysis['critical_violations']:
        response += f" CRITICAL: {'; '.join(analysis['critical_violations'])}"
    elif analysis['warnings']:
        response += f" WARNINGS: {len(analysis['warnings'])} potential issues detected."
    else:
        response += " Pipeline appears well-configured."
    
    print(f"✅ Generated enhanced intelligent response ({len(response)} chars)")
    return response


def analyze_pipeline_content(pipeline_text: str, rules: list) -> dict:
    """Intelligently analyze pipeline against rules"""
    pipeline_lower = pipeline_text.lower()
    
    analysis = {
        'evaluated_rules': [],
        'satisfied_count': 0,
        'violated_count': 0,
        'risk_factors': [],
        'missing_stages': [],
        'present_stages': []
    }
    
    # Detect actual pipeline stages
    stages_present = []
    critical_stages = ['clone', 'prepare', 'configure', 'build']
    
    for stage in critical_stages:
        if stage in pipeline_lower:
            stages_present.append(stage)
        else:
            analysis['missing_stages'].append(f"Missing {stage} stage")
    
    analysis['present_stages'] = stages_present
    
    # Analyze each rule against pipeline
    for rule in rules:
        rule_satisfied = True
        
        # Disk space rule
        if "disk space" in rule.lower():
            if not any(term in pipeline_lower for term in ['sstate_dir', 'dl_dir', 'tmpdir']):
                rule_satisfied = False
                analysis['risk_factors'].append("No disk space configuration found")
        
        # Build tools rule
        elif "build tool" in rule.lower():
            if "bitbake" not in pipeline_lower:
                rule_satisfied = False
                analysis['risk_factors'].append("BitBake build tool not found in pipeline")
        
        # Environment rule
        elif "environment" in rule.lower():
            if not any(term in pipeline_lower for term in ['environment', 'bb_number_threads', 'parallel_make']):
                rule_satisfied = False
                analysis['risk_factors'].append("Build environment not properly configured")
        
        # Permission rule
        elif "permission" in rule.lower():
            if "chown" not in pipeline_lower and "chmod" not in pipeline_lower:
                rule_satisfied = False
                analysis['risk_factors'].append("No permission management found")
        
        status = "PASS" if rule_satisfied else "FAIL"
        analysis['evaluated_rules'].append(f"{rule} - {status}")
        
        if rule_satisfied:
            analysis['satisfied_count'] += 1
        else:
            analysis['violated_count'] += 1
    
    return analysis

def generate_intelligent_response(analysis: dict, technologies: list, rules: list) -> str:
    """Generate response based on actual analysis - FORCE STRUCTURED FORMAT"""
    
    # Calculate prediction based on actual analysis
    total_rules = len(rules)
    violation_ratio = analysis['violated_count'] / total_rules if total_rules > 0 else 1
    
    if violation_ratio >= 0.6 or len(analysis.get('missing_stages', [])) >= 3:
        prediction = "FAIL"
        confidence = max(20, int(80 * violation_ratio))
    elif violation_ratio >= 0.3 or len(analysis.get('missing_stages', [])) >= 1:
        prediction = "HIGH_RISK"
        confidence = max(60, int(70 * (1 - violation_ratio)))
    else:
        prediction = "PASS"
        confidence = max(80, int(90 * (1 - violation_ratio)))
    
    # FORCE STRUCTURED FORMAT - NO NARRATIVE!
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES:
"""
    
    if analysis.get('evaluated_rules'):
        for rule in analysis['evaluated_rules']:
            response += f"• {rule}\n"
    else:
        response += "• No rules found in knowledge base\n"
    
    response += """HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
✅ Proper stage sequence (Clone → Prepare → Configure → Build)
✅ Environment variables properly set
FAILURE PATTERNS:
❌ Missing critical build stages
❌ Permission issues during build
APPLICABLE_RULES: """ + str(total_rules) + """
SATISFIED_RULES: """ + str(analysis.get('satisfied_count', 0)) + """
VIOLATED_RULES: """ + str(analysis.get('violated_count', 0)) + """
RISK_FACTORS: """ + ('; '.join(analysis.get('risk_factors', []) + analysis.get('missing_stages', [])) if analysis.get('risk_factors') or analysis.get('missing_stages') else 'None') + """
PREDICTION: """ + prediction + """
CONFIDENCE: """ + str(confidence) + """%
REASONING: Pipeline analysis shows """ + str(len(analysis.get('present_stages', []))) + """ out of 4 critical stages present. """
    
    if analysis.get('violated_count', 0) > 0:
        response += f"{analysis['violated_count']} rule violations detected. "
    
    if analysis.get('missing_stages'):
        response += f"Missing stages: {', '.join(analysis['missing_stages'])}. "
    
    if prediction == "FAIL":
        response += "Critical issues require attention before build."
    elif prediction == "HIGH_RISK":
        response += "Some issues detected, proceed with caution."
    else:
        response += "Pipeline appears properly configured."
    
    print(f" RETURNING STRUCTURED RESPONSE: {response[:200]}")
    return response



def generate_structured_response(rules, query, evaluation, technologies, patterns):
    """Build the exact structured output WITH CITATIONS"""
    response = f"""DETECTED_STACK: {', '.join(technologies)}
ESTABLISHED_RULES WITH CITATIONS:
"""
    
    # ✅ FIXED: Only ONE loop through rules (not two!)
    if rules:  # Use rules (has citations), not evaluation['evaluated_rules']
        for rule in rules[:30]:
            # Extract rule information
            if isinstance(rule, dict):
                rule_text = rule.get('rule_text', str(rule))[:150]  # Increased to 150 chars
                confidence = rule.get('confidence', 0.8)
                
                metadata = rule.get('metadata', {})
                citation = build_source_citation(metadata) if metadata else ''
            else:
                rule_text = str(rule)[:150]
                confidence = 0.8
                citation = ''
            
            # Print rule with confidence
            response += f"• {rule_text}... - PASS (conf: {confidence})\n"
            
            #  NEW: Add citation if available
            if citation:
                response += f"   {citation}\n"
    else:
        response += "• No rules extracted from knowledge base\n"
    
    
    response += """
HISTORICAL_PATTERNS:
SUCCESS PATTERNS:
"""
    success_patterns = list(set(patterns['success_indicators']))[:3]
    if success_patterns:
        for pattern in success_patterns:
            if pattern.strip():
                response += f"✅ {pattern.strip()}\n"
    else:
        response += "✅ No success patterns found in logs\n"
    
    response += """FAILURE PATTERNS:
"""
    failure_patterns = list(set(patterns['failure_indicators']))[:3]
    if failure_patterns:
        for pattern in failure_patterns:
            if pattern.strip():
                response += f"❌ {pattern.strip()}\n"
    else:
        response += "❌ No failure patterns in logs\n"
    
    response += f"""APPLICABLE_RULES: {len(rules)}
SATISFIED_RULES: {evaluation['satisfied_count']}
VIOLATED_RULES: {evaluation['violated_count']}
RISK_FACTORS: {'; '.join(evaluation['risk_factors']) if evaluation['risk_factors'] else 'None'}
PREDICTION: {evaluation['prediction']}
CONFIDENCE: {evaluation['confidence']}%
REASONING: Analysis based on {len(rules)} rules from knowledge base."""
    
    if evaluation['violated_count'] > 0:
        response += f" {evaluation['violated_count']} violations detected."
    
    if evaluation['prediction'] == "FAIL":
        response += " Multiple critical requirements not met."
    elif evaluation['prediction'] == "HIGH_RISK":
        response += " Some requirements missing, proceed with caution."
    else:
        response += " Configuration appears compliant."
    
    print(f"✅ Generated structured response: {len(response)} characters")
    return response


# def build_confluence_context(confluence_docs):
#     """Build context from confluence rules and guidelines"""
#     if not confluence_docs:
#         return "No build guidelines found in knowledge base"
    
#     context_parts = []
#     for doc in confluence_docs[:8]:  # Top 8 most relevant
#         source = doc.metadata.get('source', 'unknown')
#         content = doc.page_content[:500].strip()
        
#         # Only include if it looks like rules/requirements
#         if any(keyword in content.lower() for keyword in [
#             'recipe:', 'layer:', 'bitbake', 'yocto', 'build fails',
#             'dependency', 'configuration', 'jenkins', 'pipeline',
#             'disk space', 'memory', 'parallel', 'sstate'
#         ]):
#             context_parts.append(f"FROM {source}:\n{content}")
    
#     return "\n\n".join(context_parts) if context_parts else "No clear rules found in confluence documents"


def build_log_context(log_docs):
    """Build log context with specific pattern examples"""
    if not log_docs:
        return "No execution logs found in knowledge base"
    
    context_parts = []
    success_patterns = []
    failure_patterns = []
    
    for doc in log_docs[:8]:
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content[:600].strip()
        
        print(f" Intelligently analyzing log from {source}, content length: {len(content)}")
        
        # Extract both success and failure patterns intelligently
        patterns = extract_success_failure_patterns(content)
        if patterns['success_indicators']:
            success_patterns.extend(patterns['success_indicators'])
        if patterns['failure_indicators']:
            failure_patterns.extend(patterns['failure_indicators'])
        
        # Include logs that show execution patterns
        context_parts.append(f"LOG FROM {source}:\n{content}")
    
    # Build comprehensive summary with SPECIFIC EXAMPLES
    summary_parts = []
    
    if success_patterns:
        summary_parts.append("SUCCESS PATTERNS FROM LOGS:")
        for pattern in success_patterns[:3]:  # Limit for clarity
            summary_parts.append(f"LOG SUCCESS: {pattern}")
    
    if failure_patterns:
        summary_parts.append("\nFAILURE PATTERNS FROM LOGS:")
        for pattern in failure_patterns[:3]:  # Limit for clarity
            summary_parts.append(f"LOG FAILURE: {pattern}")
    
    if summary_parts:
        context_parts.insert(0, "\n".join(summary_parts))
        print(f"✅ Generated log analysis with {len(success_patterns + failure_patterns)} patterns")
    
    return "\n\n".join(context_parts) if context_parts else "No relevant execution patterns found in logs"


def build_adaptive_rule_context(rule_docs, query):
    """Build context that adapts to the detected technology stack"""
    
    # Detect technology from query
    detected_technologies = detect_technology_stack(query)
    
    # Prioritize relevant rules for detected technologies
    relevant_docs = []
    general_docs = []
    
    for doc in rule_docs[:20]:  # Analyze more docs for better coverage
        content = doc.page_content.lower()
        
        # Check if doc is relevant to detected technologies
        if any(tech.lower() in content for tech in detected_technologies):
            relevant_docs.append(doc)
        else:
            general_docs.append(doc)
    
    # Combine relevant + general rules
    combined_docs = relevant_docs[:10] + general_docs[:5]  # Prioritize relevant
    
    return "\n\n".join([doc.page_content[:400] for doc in combined_docs])


def build_pattern_context(log_docs, query):
    """Build context focusing on execution patterns relevant to the workflow"""
    
    # Extract technology-relevant log patterns
    relevant_logs = []
    
    for doc in log_docs[:15]:
        content = doc.page_content
        # Include logs that might contain relevant patterns
        if any(keyword in content.lower() for keyword in [
            'error', 'failed', 'success', 'passed', 'build', 'test', 
            'install', 'dependency', 'timeout', 'configuration'
        ]):
            relevant_logs.append(doc)
    
    return "\n\n".join([doc.page_content[:300] for doc in relevant_logs[:10]])

def detect_technology_stack(query):
    """Detect Yocto layers, Jenkins tools, and build technologies"""
    technologies = []
    
    query_lower = query.lower()
    
    # Yocto-specific patterns
    yocto_patterns = {
        'yocto-core': ['bitbake', 'yocto', 'openembedded', 'poky'],
        'meta-layers': ['meta-', 'bblayers.conf', 'layer.conf'],
        'jenkins': ['pipeline', 'jenkinsfile', 'stage', 'steps'],
        'kas': ['kas.yml', 'kas build', 'kas-container'],
        'docker': ['docker', 'container', 'dockerfile'],
        'cross-compile': ['gcc-cross', 'sysroot', 'toolchain'],
        'kernel': ['linux-yocto', 'kernel', 'defconfig'],
        'bootloader': ['u-boot', 'grub', 'systemd-boot'],
        'qt5': ['meta-qt5', 'qtbase', 'qtquick'],
        'multimedia': ['gstreamer', 'alsa', 'pulseaudio']
    }
    
    for tech, patterns in yocto_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            technologies.append(tech)
    
    # Default to general embedded if no specific tech detected
    if not technologies:
        technologies = ['general-embedded']
    
    return technologies

# def detect_technology_stack(query):
#     """Detect programming languages and technologies from workflow"""
#     technologies = []
    
#     query_lower = query.lower()
    
#     # Programming languages and frameworks
#     tech_patterns = {
#         'node.js': ['node', 'npm', 'yarn', 'package.json', 'javascript'],
#         'python': ['python', 'pip', 'requirements.txt', 'pytest', 'django', 'flask'],
#         'java': ['java', 'maven', 'gradle', 'pom.xml', 'junit', 'spring'],
#         'dotnet': ['.net', 'dotnet', 'nuget', 'csharp', 'msbuild'],
#         'go': ['go', 'golang', 'go.mod', 'go.sum'],
#         'rust': ['rust', 'cargo', 'cargo.toml'],
#         'php': ['php', 'composer', 'composer.json', 'phpunit'],
#         'ruby': ['ruby', 'gem', 'bundler', 'gemfile', 'rails'],
#         'docker': ['docker', 'dockerfile', 'container'],
#         'kubernetes': ['kubectl', 'k8s', 'kubernetes', 'helm'],
#     }
    
#     for tech, patterns in tech_patterns.items():
#         if any(pattern in query_lower for pattern in patterns):
#             technologies.append(tech)
    
#     # Default to general CI/CD if no specific tech detected
#     if not technologies:
#         technologies = ['general-cicd']
    
#     return technologies


# def perform_adaptive_analysis(query):
#     """Platform-agnostic analysis that returns structured format"""
#     technologies = detect_technology_stack(query)
    
#     # Instead of calling LLM, return structured format directly
#     return f"""DETECTED_STACK: {', '.join(technologies)}
# ESTABLISHED_RULES:
# • No adaptive rules available - PASS
# HISTORICAL_PATTERNS:
# SUCCESS PATTERNS:
# ✅ No historical patterns available
# FAILURE PATTERNS:
# ❌ No historical patterns available
# APPLICABLE_RULES: 0
# SATISFIED_RULES: 0
# VIOLATED_RULES: 0
# RISK_FACTORS: None
# PREDICTION: PASS/FAIL
# CONFIDENCE: 50%
# REASONING: Adaptive analysis mode - no specific rules available."""




async def get_rag_response_async(query: str, user_id: str) -> str:
    # Force check and refresh from MinIO before EVERY query if enabled
    if CHECK_MINIO_ON_EVERY_QUERY:
        print("Checking MinIO for updates before query...")
        await refresh_vector_store_if_needed(force_refresh=True)
    
    user_lock = await get_user_lock(user_id)
    
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    async with user_lock:
        print(f"Processing request for user: {user_id}")
        response = await run_in_threadpool(sync_get_rag_response, query)
        print(f"Completed request for user: {user_id}")
        return response



# ============= WORKSPACE FILE PROCESSING FUNCTIONS =============
def load_workspace_processed_state() -> Dict[str, str]:
    """Load state of processed workspace files"""
    if os.path.exists(WORKSPACE_STATE_PATH) and not FORCE_WORKSPACE_REBUILD:
        try:
            with open(WORKSPACE_STATE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading workspace state: {e}")
    return {}

def save_workspace_processed_state(state: Dict[str, str]):
    """Save state of processed workspace files"""
    with open(WORKSPACE_STATE_PATH, "wb") as f:
        pickle.dump(state, f)

def file_hash(path: str) -> str:
    """Calculate SHA-256 hash of file"""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"Error hashing file {path}: {e}")
        return ""

def discover_workspace_files() -> list[str]:
    """Discover relevant files in workspace"""
    file_paths = []
    if not os.path.exists(WORKSPACE_DIR):
        print(f"⚠️ Workspace directory not found: {WORKSPACE_DIR}")
        return file_paths
        
    print(f"🔍 Scanning workspace: {WORKSPACE_DIR}")
    for root, _, files in os.walk(WORKSPACE_DIR):
        for fn in files:
            if (fn == "Jenkinsfile" or 
                fn.endswith(('.bb', '.conf', '.bbappend', '.inc')) or
                fn in ('layer.conf', 'local.conf', 'bblayers.conf')):
                full_path = os.path.join(root, fn)
                file_paths.append(full_path)
                print(f"   Found: {os.path.relpath(full_path, WORKSPACE_DIR)}")
    
    print(f"📊 Total workspace files discovered: {len(file_paths)}")
    return file_paths

def process_workspace_files_incrementally():
    """Process workspace files and add to existing vector store"""
    global vectorstore
    
    print("🔍 Processing workspace files incrementally...")
    
    # Discover files
    file_paths = discover_workspace_files()
    if not file_paths:
        print("⚠️ No relevant workspace files found")
        return 0
    
    # Load processed state
    processed_state = load_workspace_processed_state()
    new_state = processed_state.copy()
    changed_files = []
    
    # Identify changed files
    for path in file_paths:
        try:
            if not os.path.exists(path):
                continue
                
            current_hash = file_hash(path)
            if not current_hash:
                continue
                
            if path not in processed_state or processed_state[path] != current_hash:
                changed_files.append(path)
                new_state[path] = current_hash
        except Exception as e:
            print(f"❌ Error checking {path}: {e}")
            continue
    
    # First run or forced rebuild
    if not processed_state or FORCE_WORKSPACE_REBUILD:
        changed_files = [p for p in file_paths if os.path.exists(p)]
        new_state = {}
        for path in changed_files:
            hash_val = file_hash(path)
            if hash_val:
                new_state[path] = hash_val
    
    if not changed_files:
        print("✅ No workspace file changes detected")
        return 0
    
    print(f"📝 Processing {len(changed_files)} changed workspace files")
    
    # Process changed files into documents
    workspace_docs = []
    for path in changed_files:
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()
            
            if len(content.strip()) < 10:
                continue
            
            # Determine file type
            rel_path = os.path.relpath(path, WORKSPACE_DIR)
            if path.endswith('.bb'):
                file_type = 'yocto_recipe'
            elif path.endswith(('.conf', '.inc')):
                file_type = 'yocto_config'
            elif os.path.basename(path) == 'Jenkinsfile':
                file_type = 'jenkins_pipeline'
            else:
                file_type = 'workspace_file'
            
            # Create document with workspace metadata (integrate into same vector store)
            doc = Document(
                page_content=f"[WORKSPACE FILE: {rel_path}]\n{content}",
                metadata={
                    'source': f'workspace:{rel_path}',
                    'file_type': file_type,
                    'category': 'workspace',  # Mark as workspace content
                    'full_path': path,
                    'file_size': len(content),
                    'last_modified': time.time()
                }
            )
            workspace_docs.append(doc)
            print(f"  ✅ Prepared: {rel_path} ({file_type}, {len(content)} chars)")
            
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            continue
    
    if not workspace_docs:
        print("No valid workspace documents to index")
        return 0
    
    # Add workspace documents to existing vector store
    if vectorstore is not None:
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50
            )
            
            split_docs = []
            for doc in workspace_docs:
                chunks = text_splitter.split_documents([doc])
                split_docs.extend(chunks)
            
            # Add to existing vector store (same one used for MinIO)
            vectorstore.add_documents(split_docs)
            vectorstore.save_local(FAISS_INDEX_PATH)
            
            print(f"✅ Added {len(workspace_docs)} workspace documents ({len(split_docs)} chunks) to existing vector store")
            
        except Exception as e:
            print(f"❌ Error adding workspace docs to vector store: {e}")
            return 0
    else:
        print("⚠️ Vector store not initialized, cannot add workspace files")
        return 0
    
    # Save updated state
    save_workspace_processed_state(new_state)
    return len(workspace_docs)

# ============= ENHANCED API ENDPOINT =============



# ========= MAIN API ENDPOINTS =========



@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions endpoint"""
    api_start_time = time.time()
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail=f"User {user_id} already has a request in progress.")
    
    # Generate UUID FIRST
    prediction_id = str(uuid.uuid4())
    print(f"Generated prediction_id: {prediction_id}")
    
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = request.messages[-1].get("content", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message content")
        
        created_timestamp = int(time.time())
        completion_id = f"chatcmpl-{created_timestamp}-{hash(user_message) % 10000:04d}"
        
        if request.model in ["codellama:7b", "codellama-rag"]:
            raw_response = await get_rag_response_async(user_message, user_id)
        else:
            raw_response = f"Model '{request.model}' not supported"
        
        sanitized_response = sanitize_response_content(raw_response)
        
        # Store prediction
        try:
            if connection_pool:
                script_hash = hashlib.sha256(user_message.encode()).hexdigest()
                
                predicted_result = "UNKNOWN"
                confidence_score = 50
                violated_rules = 0
                
                if "PREDICTION:" in sanitized_response:
                    for line in sanitized_response.split('\n'):
                        if 'PREDICTION:' in line:
                            pred_text = line.split('PREDICTION:')[1].strip()
                            if 'FAIL' in pred_text:
                                predicted_result = 'FAIL'
                            elif 'PASS' in pred_text:
                                predicted_result = 'PASS'
                            elif 'HIGH-RISK' in pred_text:
                                predicted_result = 'HIGH-RISK'
                            break
                
                if "CONFIDENCE:" in sanitized_response:
                    for line in sanitized_response.split('\n'):
                        if 'CONFIDENCE:' in line:
                            conf_text = ''.join(filter(str.isdigit, line))
                            if conf_text:
                                confidence_score = int(conf_text)
                            break
                
                if "VIOLATED_RULES:" in sanitized_response:
                    for line in sanitized_response.split('\n'):
                        if 'VIOLATED_RULES:' in line:
                            viol_text = ''.join(filter(str.isdigit, line))
                            if viol_text:
                                violated_rules = int(viol_text)
                            break
                
                with get_db_connection() as conn:
                    with get_db_cursor(conn) as cur:
                        cur.execute("""
                            INSERT INTO predictions (
                                user_id, pipeline_name, predicted_result,
                                confidence_score, violated_rules,
                                pipeline_script_hash, detected_stack, rules_applied, id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            user_id, "chat_query", predicted_result,
                            confidence_score, violated_rules,
                            script_hash, ['jenkins', 'yocto'],
                            json.dumps([]), prediction_id
                        ))
                        print(f"✅ Stored prediction {prediction_id} in DB")
        
        except Exception as db_error:
            print(f"⚠ DB error: {db_error}")
        
        processing_time = time.time() - api_start_time
        print(f"API response time: {processing_time:.2f}s")
        print(f"📤 Returning prediction_id: {prediction_id}")
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": request.model,
            "system_fingerprint": "fp_rag_system",
            "prediction_id": prediction_id,  # RETURN THE UUID
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": sanitized_response},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(sanitized_response.split()),
                "total_tokens": len(user_message.split()) + len(sanitized_response.split())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "id": f"chatcmpl-error-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_rag_system",
            "prediction_id": prediction_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }




@app.post("/minio/webhook")
async def minio_webhook(request: Request):
    """Handle MinIO bucket notifications - ignore access events"""
    try:
        event_data = await request.json()
        print(f"Received MinIO webhook notification")
        
        if "Records" not in event_data:
            return {"status": "ignored", "message": "No records in event"}
        
        should_rebuild = False
        processed_events = []
        
        for record in event_data["Records"]:
            event_name = record.get("eventName", "")
            bucket_name = record.get("s3", {}).get("bucket", {}).get("name", "")
            object_key = record.get("s3", {}).get("object", {}).get("key", "")
            
            # FILTER: Only process file creation/deletion events
            if not (event_name.startswith("s3:ObjectCreated") or event_name.startswith("s3:ObjectRemoved")):
                print(f"Ignoring access event: {event_name}")
                continue
                
            # Only process events from our target bucket
            if bucket_name != MINIO_BUCKET:
                continue
                
            should_rebuild = True
            processed_events.append({
                "event": event_name,
                "object": object_key,
                "bucket": bucket_name
            })
            print(f"Processing relevant event: {event_name} on {object_key}")
        
        if should_rebuild:
            print(f"Rebuilding vector store due to {len(processed_events)} file events...")
            asyncio.create_task(refresh_vector_store_if_needed(force_refresh=True))
            
            return {
                "status": "rebuilding", 
                "message": f"Triggered rebuild for {len(processed_events)} events",
                "events": processed_events
            }
        else:
            return {"status": "ignored", "message": "No relevant file change events"}
            
    except Exception as e:
        print(f"❌ Error in MinIO webhook: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/validate-pipeline")
async def validate_pipeline():
    """Enhanced pipeline validation with Yocto file analysis and confidence suggestions"""
    try:
        print("🚀 Starting ENHANCED pipeline validation with Yocto analysis...")
        
        # Discover workspace files
        workspace_files = []
        workspace_dir = "/var/jenkins_home/workspace/Yocto-Build-Pipeline"
        yocto_file_paths = []
        
        if os.path.exists(workspace_dir):
            print(f"🔍 Scanning workspace: {workspace_dir}")
            for root, _, files in os.walk(workspace_dir):
                for fn in files:
                    full_path = os.path.join(root, fn)
                    rel_path = os.path.relpath(full_path, workspace_dir)
                    
                    if (fn == "Jenkinsfile" or 
                        fn.endswith(('.bb', '.conf', '.bbappend', '.inc', '.bbclass')) or
                        fn in ('layer.conf', 'local.conf', 'bblayers.conf')):
                        workspace_files.append(rel_path)
                        
                        # Collect Yocto files for analysis
                        if fn.endswith(('.bb', '.conf', '.bbappend', '.inc', '.bbclass')) or fn in ('layer.conf', 'local.conf', 'bblayers.conf'):
                            yocto_file_paths.append(full_path)
                            
            print(f" Found {len(workspace_files)} workspace files")
            print(f" Found {len(yocto_file_paths)} Yocto files for analysis")
        else:
            print(f"⚠️ Workspace directory not found: {workspace_dir}")
        
        # Read Jenkinsfile
        jenkinsfile_path = os.path.join(workspace_dir, "Jenkinsfile")
        jenkinsfile_content = "No Jenkinsfile found in workspace"
        
        if os.path.exists(jenkinsfile_path):
            with open(jenkinsfile_path, "r", errors="ignore") as f:
                jenkinsfile_content = f.read()
            print("✅ Jenkinsfile found and read")
        
        # **NEW: YOCTO-SPECIFIC ANALYSIS**
        print("Analyzing Yocto workspace files...")
        yocto_specific_rules = []
        if yocto_file_paths:
            yocto_specific_rules = extract_yocto_specific_rules(yocto_file_paths)
            print(f"✅ Extracted {len(yocto_specific_rules)} Yocto-specific rules")
            
            for rule in yocto_specific_rules[:5]:  # Show first 5 for debugging
                print(f"   📋 {rule['rule_type']}: {rule['rule_text'][:80]}...")
        
        # **NEW: BUILD YOCTO CONTEXT FOR LLM**
        yocto_context = ""
        if yocto_specific_rules:
            yocto_dependencies = [r for r in yocto_specific_rules if 'DEPENDENCY' in r['rule_type']]
            yocto_configs = [r for r in yocto_specific_rules if 'CONFIG' in r['rule_type'] or 'THREADS' in r['rule_type']]
            yocto_sources = [r for r in yocto_specific_rules if 'SOURCE' in r['rule_type']]
            
            yocto_context = f"""
 YOCTO WORKSPACE ANALYSIS:
Found {len(yocto_specific_rules)} specific rules from {len(yocto_file_paths)} Yocto files:

RECIPE DEPENDENCIES ({len(yocto_dependencies)}):
{chr(10).join([f"• {r['rule_text']}" for r in yocto_dependencies[:8]])}

BUILD CONFIGURATION ({len(yocto_configs)}):
{chr(10).join([f"• {r['rule_text']}" for r in yocto_configs[:5]])}

SOURCE REQUIREMENTS ({len(yocto_sources)}):
{chr(10).join([f"• {r['rule_text']}" for r in yocto_sources[:5]])}

YOCTO FILES ANALYZED:
{chr(10).join([f"• {os.path.basename(f)}" for f in yocto_file_paths[:10]])}
"""
        
        # Enhanced prompt with Yocto analysis
        enhanced_prompt = f"""=== JENKINS PIPELINE ===
{jenkinsfile_content}

=== WORKSPACE CONTEXT ===
Workspace Directory: {workspace_dir}
Files Detected: {', '.join(workspace_files) if workspace_files else 'None'}
Total Files: {len(workspace_files)}

{yocto_context}

=== COMPREHENSIVE ANALYSIS REQUEST ===
Analyze this Jenkins pipeline against:
1. Knowledge base rules (from Confluence/logs)
2. Workspace Yocto files analysis above
3. Historical patterns

**CRITICAL**: If confidence is below 90%, provide specific suggestions for improvement.

Evaluate BOTH generic CI/CD rules AND Yocto-specific requirements.
Consider recipe dependencies, build configuration, and source requirements.

OUTPUT FORMAT: Use your standard structured format with YOCTO-specific insights."""

        print(f"📝 Enhanced prompt created ({len(enhanced_prompt)} chars)")
        
        # Get analysis from RAG system
        user_lock = await get_user_lock("pipeline-validator")
        async with user_lock:
            print("🧠 Calling enhanced RAG analysis...")
            analysis_result = await run_in_threadpool(
                sync_get_rag_response,
                enhanced_prompt
            )
        
        # **NEW: CONFIDENCE ENHANCEMENT WITH SUGGESTIONS**
        confidence_suggestions = []
        confidence_boost = 0
        
        if yocto_specific_rules:
            # Analyze Yocto completeness
            has_machine_config = any('MACHINE_TARGET' in r['rule_type'] for r in yocto_specific_rules)
            has_build_deps = any('DEPENDENCY' in r['rule_type'] for r in yocto_specific_rules)
            has_threads_config = any('THREADS' in r['rule_type'] for r in yocto_specific_rules)
            has_distro_config = any('DISTRO' in r['rule_type'] for r in yocto_specific_rules)
            
            if not has_machine_config:
                confidence_suggestions.append({
                    "issue": "Missing MACHINE configuration",
                    "fix": "Add MACHINE = 'qemux86-64' to local.conf",
                    "impact": "15% confidence reduction",
                    "priority": "HIGH"
                })
            else:
                confidence_boost += 5
            
            if not has_threads_config:
                confidence_suggestions.append({
                    "issue": "No BitBake thread optimization",
                    "fix": "Add BB_NUMBER_THREADS = '8' to local.conf for faster builds",
                    "impact": "8% confidence reduction", 
                    "priority": "MEDIUM"
                })
            else:
                confidence_boost += 3
                
            if not has_distro_config:
                confidence_suggestions.append({
                    "issue": "Missing DISTRO configuration",
                    "fix": "Add DISTRO = 'poky' or specific distro to local.conf",
                    "impact": "10% confidence reduction",
                    "priority": "MEDIUM"
                })
            else:
                confidence_boost += 3
        
        # Add general Jenkins suggestions
        jenkins_lower = jenkinsfile_content.lower()
        if 'timeout' not in jenkins_lower:
            confidence_suggestions.append({
                "issue": "No pipeline timeout configured",
                "fix": "Add timeout(time: 6, unit: 'HOURS') around pipeline stages",
                "impact": "12% confidence reduction",
                "priority": "HIGH"
            })
            
        if 'agent' not in jenkins_lower or 'yocto' not in jenkins_lower:
            confidence_suggestions.append({
                "issue": "Build agent not optimized for Yocto",
                "fix": "Use agent { label 'yocto-build-node' } for dedicated Yocto environment",
                "impact": "10% confidence reduction", 
                "priority": "MEDIUM"
            })
        
        print("✅ Enhanced analysis completed with Yocto insights")
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "workspace_files_detected": len(workspace_files),
            "workspace_files": workspace_files,
            "yocto_files_analyzed": len(yocto_file_paths),
            "yocto_specific_rules": len(yocto_specific_rules),
            "yocto_insights": {
                "dependencies_found": len([r for r in yocto_specific_rules if 'DEPENDENCY' in r['rule_type']]),
                "configurations_found": len([r for r in yocto_specific_rules if 'CONFIG' in r['rule_type']]),
                "sources_found": len([r for r in yocto_specific_rules if 'SOURCE' in r['rule_type']])
            },
            "confidence_suggestions": confidence_suggestions,
            "confidence_boost_available": f"{confidence_boost}% additional confidence possible",
            "workspace_directory": workspace_dir,
            "jenkinsfile_found": os.path.exists(jenkinsfile_path),
            "enhanced_analysis": True,
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        print(f"❌ Error in enhanced validate-pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat_endpoint(
    request: dict,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Handle Cline extension requests to /api/chat"""
    try:
        print(f"Cline API request: {request}")
        
        chat_request = ChatRequest(
            messages=request.get("messages", []),
            model=request.get("model", "codellama:7b"),
            temperature=request.get("temperature", 0.4),
            max_tokens=request.get("max_tokens", 2000)
        )
        
        response = await chat_completions(chat_request, http_request, x_user_id)
        return response
        
    except Exception as e:
        print(f"❌ Error in api_chat_endpoint: {e}")
        return {
            "id": f"chatcmpl-error-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "codellama:7b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


@app.post("/api/generate")
async def ollama_generate(
    request: dict,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Ollama-compatible generate endpoint"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress. Please wait for completion."
        )
    
    try:
        print(f"Ollama generate request from {user_id}: {request}")
        
        prompt = request.get("prompt", "")
        model = request.get("model", "codellama:7b")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        print(f"Prompt: {prompt}")
        print(f"Model: {model}")
        
        if model in ["codellama:7b", "codellama-rag"]:
            response_text = await get_rag_response_async(prompt, user_id)
        else:
            response_text = f"Model '{model}' not supported. Available models: codellama:7b, codellama-rag"
        
        sanitized_response = sanitize_response_content(response_text)
        
        result = {
            "model": model,
            "created_at": "2024-01-01T00:00:00Z",
            "response": sanitized_response,
            "done": True,
            "context": [],
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": len(prompt.split()),
            "prompt_eval_duration": 200000000,
            "eval_count": len(sanitized_response.split()),
            "eval_duration": 700000000
        }
        
        print(f"✅ Generated response for {user_id}: {sanitized_response[:100]}...")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in ollama_generate for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible completions endpoint"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress. Please wait for completion."
        )
    
    try:
        if request.model in ["codellama:7b", "codellama-rag"]:
            response = await get_rag_response_async(request.prompt, user_id)
        else:
            response = f"Model '{request.model}' not supported"
            
        sanitized_response = sanitize_response_content(response)
        
        return CompletionResponse(
            choices=[{
                "index": 0,
                "text": sanitized_response,
                "finish_reason": "stop"
            }],
            model=request.model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== Feedback Endpts====== #
# @app.post("/api/feedback/submit")
# async def submit_feedback(feedback: FeedbackRequest, x_user_id: Optional[str] = Header(None)):
#     """Submit feedback on a prediction"""
#     user_id = x_user_id or "anonymous"
    
#     try:
#         with get_db_connection() as conn:
#             with get_db_cursor(conn) as cur:
#                 # Get original prediction
#                 cur.execute("""
#                     SELECT predicted_result, rules_applied
#                     FROM predictions WHERE id = %s
#                 """, (feedback.prediction_id,))
                
#                 prediction = cur.fetchone()
#                 if not prediction:
#                     raise HTTPException(status_code=404, detail="Prediction not found")
                
#                 # Check if correct
#                 correct = (
#                     (prediction['predicted_result'] == "PASS" and feedback.actual_build_result == "SUCCESS") or
#                     (prediction['predicted_result'] == "FAIL" and feedback.actual_build_result == "FAILURE")
#                 )
                
#                 # Insert feedback
#                 cur.execute("""
#                     INSERT INTO feedback (
#                         prediction_id, user_id, actual_build_result,
#                         correct_prediction, corrected_confidence,
#                         missed_issues, false_positives, user_comments,
#                         suggested_rules, feedback_type
#                     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                     RETURNING id
#                 """, (
#                     feedback.prediction_id,
#                     user_id,
#                     feedback.actual_build_result,
#                     correct,
#                     feedback.corrected_confidence,
#                     feedback.missed_issues,
#                     feedback.false_positives,
#                     feedback.user_comments,
#                     json.dumps(feedback.suggested_rules) if feedback.suggested_rules else None,
#                     'manual'
#                 ))
                
#                 feedback_id = cur.fetchone()['id']
                
#                 # Update prediction
#                 cur.execute("""
#                     UPDATE predictions
#                     SET actual_result = %s, feedback_received_at = NOW()
#                     WHERE id = %s
#                 """, (feedback.actual_build_result, feedback.prediction_id))
                
#                 # Learn from feedback async
#                 await learn_from_feedback_async(str(feedback_id), prediction['rules_applied'], feedback, correct)
                
#                 return {
#                     "status": "success",
#                     "feedback_id": str(feedback_id),
#                     "was_correct": correct,
#                     "learning_applied": True
#                 }
                
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error submitting feedback: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback/submit")
async def submit_feedback(feedback: FeedbackRequest, x_user_id: Optional[str] = Header(None)):
    """Submit feedback on a prediction"""
    user_id = x_user_id or "anonymous"
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT predicted_result, rules_applied
                    FROM predictions WHERE id::text = %s
                """, (feedback.prediction_id,))
                
                prediction = cur.fetchone()
                if not prediction:
                    raise HTTPException(status_code=404, detail="Prediction not found")
                
                correct = (
                    (prediction['predicted_result'] == "PASS" and feedback.actual_build_result == "SUCCESS") or
                    (prediction['predicted_result'] == "FAIL" and feedback.actual_build_result == "FAILURE")
                )
                
                # Convert to Python lists (NOT JSON strings)
                missed = feedback.missed_issues if feedback.missed_issues else []
                false_pos = feedback.false_positives if feedback.false_positives else []
                
                cur.execute("""
                    INSERT INTO feedback (
                        prediction_id, user_id, actual_build_result,
                        correct_prediction, corrected_confidence,
                        missed_issues, false_positives, user_comments,
                        feedback_type
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    feedback.prediction_id,
                    user_id,
                    feedback.actual_build_result,
                    correct,
                    feedback.corrected_confidence,
                    missed,  # Python list - psycopg2 converts to PostgreSQL array
                    false_pos,
                    feedback.user_comments,
                    feedback.feedback_type
                ))
                
                feedback_id = cur.fetchone()['id']
                
                cur.execute("""
                    UPDATE predictions
                    SET actual_result = %s, feedback_received_at = NOW()
                    WHERE id = %s::uuid
                """, (feedback.actual_build_result, feedback.prediction_id))
                
                conn.commit()
                print(f"✅ Feedback {feedback_id} committed")
        
        # Learn from feedback
        try:
            await learn_from_feedback_async(str(feedback_id), prediction['rules_applied'], feedback, correct)
        except Exception as e:
            print(f"⚠️ Learning error: {e}")
        
        return {
            "status": "success",
            "feedback_id": str(feedback_id),
            "was_correct": correct
        }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def learn_from_feedback_async(feedback_id: str, rules_applied: str, feedback: FeedbackRequest, was_correct: bool):
    """Learn from feedback asynchronously"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Verify feedback exists first
                cur.execute("SELECT id FROM feedback WHERE id = %s::uuid", (feedback_id,))
                if not cur.fetchone():
                    print(f"⚠️ Feedback {feedback_id} not found in DB yet, skipping learning")
                    return
                
                rules = json.loads(rules_applied) if isinstance(rules_applied, str) else rules_applied
                
                # Update rule performance
                for rule in rules[:20]:
                    rule_text = rule.get('rule_text', '') if isinstance(rule, dict) else str(rule)
                    if not rule_text or len(rule_text) < 10:
                        continue
                    
                    cur.execute("""
                        INSERT INTO rule_performance (
                            rule_text, rule_type, total_applications, correct_predictions
                        ) VALUES (%s, %s, 1, %s)
                        ON CONFLICT (rule_text) DO UPDATE SET
                            total_applications = rule_performance.total_applications + 1,
                            correct_predictions = rule_performance.correct_predictions + EXCLUDED.correct_predictions,
                            last_updated = NOW()
                    """, (
                        rule_text[:500], 
                        rule.get('rule_type', 'unknown') if isinstance(rule, dict) else 'unknown', 
                        1 if was_correct else 0
                    ))
                
                # Learn from missed issues
                if feedback.missed_issues:
                    for issue in feedback.missed_issues[:10]:
                        if issue and len(issue.strip()) > 5:
                            cur.execute("""
                                INSERT INTO learned_patterns (
                                    pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
                                ) VALUES ('failure_indicator', %s, %s::uuid, 0.15)
                                ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
                                    occurrences = learned_patterns.occurrences + 1,
                                    confidence_boost = LEAST(learned_patterns.confidence_boost + 0.05, 0.3)
                            """, (issue[:500], feedback_id))
                
                # Learn from false positives
                if feedback.false_positives:
                    for fp in feedback.false_positives[:10]:
                        if fp and len(fp.strip()) > 5:
                            cur.execute("""
                                INSERT INTO learned_patterns (
                                    pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
                                ) VALUES ('false_positive', %s, %s::uuid, -0.1)
                                ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
                                    occurrences = learned_patterns.occurrences + 1,
                                    confidence_boost = GREATEST(learned_patterns.confidence_boost - 0.05, -0.2)
                            """, (fp[:500], feedback_id))
                
                conn.commit()
                print(f"✅ Learned from feedback {feedback_id}")
                
    except Exception as e:
        print(f"Error learning from feedback: {e}")
        import traceback
        traceback.print_exc()


# async def learn_from_feedback_async(feedback_id: str, rules_applied: str, feedback: FeedbackRequest, was_correct: bool):
#     """Learn from feedback asynchronously"""
#     try:
#         with get_db_connection() as conn:
#             with get_db_cursor(conn) as cur:
#                 rules = json.loads(rules_applied) if isinstance(rules_applied, str) else rules_applied
                
#                 # Update rule performance
#                 for rule in rules[:20]:  # Limit to first 20 rules
#                     rule_text = rule.get('rule_text', '') if isinstance(rule, dict) else str(rule)
#                     if not rule_text or len(rule_text) < 10:
#                         continue
                    
#                     cur.execute("""
#                         INSERT INTO rule_performance (
#                             rule_text, rule_type, total_applications, correct_predictions
#                         ) VALUES (%s, %s, 1, %s)
#                         ON CONFLICT (rule_text) DO UPDATE SET
#                             total_applications = rule_performance.total_applications + 1,
#                             correct_predictions = rule_performance.correct_predictions + EXCLUDED.correct_predictions,
#                             last_updated = NOW()
#                     """, (rule_text[:500], rule.get('rule_type', 'unknown') if isinstance(rule, dict) else 'unknown', 1 if was_correct else 0))
                
#                 # Learn from missed issues
#                 for issue in feedback.missed_issues[:10]:
#                     cur.execute("""
#                         INSERT INTO learned_patterns (
#                             pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
#                         ) VALUES ('failure_indicator', %s, %s, 0.15)
#                         ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
#                             occurrences = learned_patterns.occurrences + 1,
#                             confidence_boost = LEAST(learned_patterns.confidence_boost + 0.05, 0.3)
#                     """, (issue[:500], feedback_id))
                
#                 # Learn from false positives
#                 for fp in feedback.false_positives[:10]:
#                     cur.execute("""
#                         INSERT INTO learned_patterns (
#                             pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
#                         ) VALUES ('false_positive', %s, %s, -0.1)
#                         ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
#                             occurrences = learned_patterns.occurrences + 1,
#                             confidence_boost = GREATEST(learned_patterns.confidence_boost - 0.05, -0.2)
#                     """, (fp[:500], feedback_id))
                
#                 print(f"✅ Learned from feedback {feedback_id}")
#     except Exception as e:
#         print(f"Error learning from feedback: {e}")


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Overall stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) as correct,
                        AVG(p.confidence_score) as avg_confidence
                    FROM predictions p
                    LEFT JOIN feedback f ON f.prediction_id = p.id
                    WHERE p.created_at > NOW() - INTERVAL '30 days'
                """)
                
                stats = cur.fetchone()
                
                # Get learned patterns count
                cur.execute("""
                    SELECT COUNT(*) as learned_patterns
                    FROM learned_patterns
                    WHERE occurrences >= 2
                """)
                
                patterns = cur.fetchone()
                
                return {
                    "total_predictions": stats['total_predictions'] or 0,
                    "correct_predictions": stats['correct'] or 0,
                    "accuracy_percentage": round((stats['correct'] / stats['total_predictions']) * 100, 2) if stats['total_predictions'] > 0 else 0,
                    "average_confidence": round(stats['avg_confidence'] or 0, 2),
                    "learned_patterns": patterns['learned_patterns'] or 0
                }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/feedback/list")
async def list_feedback(limit: int = 50):
    """List recent feedback with predictions"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT 
                        f.id as feedback_id,
                        f.prediction_id,
                        p.pipeline_name,
                        p.predicted_result,
                        p.confidence_score,
                        f.actual_build_result,
                        f.correct_prediction,
                        f.corrected_confidence,
                        f.missed_issues,
                        f.false_positives,
                        f.user_comments,
                        f.created_at
                    FROM feedback f
                    JOIN predictions p ON f.prediction_id = p.id
                    ORDER BY f.created_at DESC
                    LIMIT %s
                """, (limit,))
                
                feedbacks = []
                for row in cur.fetchall():
                    feedbacks.append({
                        "feedback_id": str(row['feedback_id']),
                        "prediction_id": str(row['prediction_id']),
                        "pipeline": row['pipeline_name'],
                        "ai_predicted": row['predicted_result'],
                        "ai_confidence": row['confidence_score'],
                        "actual_result": row['actual_build_result'],
                        "was_correct": row['correct_prediction'],
                        "corrected_confidence": row['corrected_confidence'],
                        "missed_issues": row['missed_issues'],
                        "false_positives": row['false_positives'],
                        "comments": row['user_comments'],
                        "submitted_at": row['created_at'].isoformat()
                    })
                
                return {"feedbacks": feedbacks, "count": len(feedbacks)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/learning/patterns")
async def get_learned_patterns():
    """Show what patterns AI has learned"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Get failure indicators learned
                cur.execute("""
                    SELECT 
                        pattern_text,
                        pattern_type,
                        occurrences,
                        confidence_boost,
                        created_at
                    FROM learned_patterns
                    WHERE pattern_type = 'failure_indicator'
                    ORDER BY occurrences DESC
                    LIMIT 20
                """)
                
                failure_patterns = []
                for row in cur.fetchall():
                    failure_patterns.append({
                        "pattern": row['pattern_text'],
                        "learned_from_feedback_count": row['occurrences'],
                        "confidence_boost": row['confidence_boost'],
                        "first_learned": row['created_at'].isoformat()
                    })
                
                # Get false positive patterns
                cur.execute("""
                    SELECT 
                        pattern_text,
                        occurrences,
                        confidence_boost
                    FROM learned_patterns
                    WHERE pattern_type = 'false_positive'
                    ORDER BY occurrences DESC
                    LIMIT 20
                """)
                
                false_patterns = []
                for row in cur.fetchall():
                    false_patterns.append({
                        "pattern": row['pattern_text'],
                        "times_corrected": row['occurrences'],
                        "confidence_penalty": row['confidence_boost']
                    })
                
                return {
                    "failure_indicators_learned": failure_patterns,
                    "false_positives_corrected": false_patterns,
                    "total_patterns_learned": len(failure_patterns) + len(false_patterns)
                }
    except Exception as e:
        return {"error": str(e)}



@app.get("/api/learning/rules")
async def get_rule_performance():
    """Show how well each rule performs"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT 
                        rule_text,
                        rule_type,
                        total_applications,
                        correct_predictions,
                        ROUND((correct_predictions::float / total_applications * 100), 2) as accuracy
                    FROM rule_performance
                    WHERE total_applications > 0
                    ORDER BY accuracy DESC
                    LIMIT 30
                """)
                
                rules = []
                for row in cur.fetchall():
                    rules.append({
                        "rule": row['rule_text'][:100],
                        "type": row['rule_type'],
                        "times_applied": row['total_applications'],
                        "times_correct": row['correct_predictions'],
                        "accuracy_percentage": float(row['accuracy'])
                    })
                
                return {"rules": rules, "count": len(rules)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/learning/accuracy-trend")
async def get_accuracy_trend():
    """Show accuracy improvement over time"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT 
                        DATE(f.created_at) as date,
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) as correct,
                        ROUND(AVG(CASE WHEN f.correct_prediction THEN 100 ELSE 0 END), 2) as accuracy
                    FROM feedback f
                    WHERE f.created_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(f.created_at)
                    ORDER BY date ASC
                """)
                
                trend = []
                for row in cur.fetchall():
                    trend.append({
                        "date": row['date'].isoformat(),
                        "predictions": row['total_feedback'],
                        "correct": row['correct'],
                        "accuracy_percent": float(row['accuracy'])
                    })
                
                return {"accuracy_trend": trend}
    except Exception as e:
        return {"error": str(e)}


# ========= GITHUB ACTIONS SPECIFIC ENDPOINT =========

class GitHubWorkflowRequest(BaseModel):
    workflow_yaml: str
    workflow_name: str = "Unknown Workflow"

@app.post("/api/github-actions/analyze")
async def github_actions_analyze(
    request: GitHubWorkflowRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Dedicated endpoint for GitHub Actions workflow analysis"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress."
        )
    
    try:
        print(f"GitHub Actions workflow analysis request from {user_id}")
        print(f"Workflow name: {request.workflow_name}")
        
        workflow_yaml = request.workflow_yaml.strip()
        
        if not workflow_yaml:
            raise HTTPException(status_code=400, detail="No workflow YAML provided")
        
        # Check if it's actually a workflow
        if not ("name:" in workflow_yaml and ("jobs:" in workflow_yaml or "steps:" in workflow_yaml)):
            raise HTTPException(
                status_code=400, 
                detail="Invalid workflow format. Please provide a complete GitHub Actions YAML workflow"
            )
        
        # Use existing RAG analysis function
        analysis_result = await get_rag_response_async(workflow_yaml, user_id)
        sanitized_response = sanitize_response_content(analysis_result)
        
        return {
            "workflow_name": request.workflow_name,
            "analysis": sanitized_response,
            "status": "completed",
            "analyzed_by": "rag-workflow-analyzer",
            "timestamp": int(time.time())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in GitHub Actions analysis for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ========= YOCTO BUILD & JENKINS PREDICTION ENDPOINT =========

class YoctoBuildRequest(BaseModel):
    jenkins_pipeline: str = ""
    yocto_recipes: str = ""
    build_config: str = ""
    target_image: str = "Unknown Target"

@app.post("/api/yocto/predict")
async def yocto_build_predict(
    request: YoctoBuildRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Predict Yocto build success/failure from Jenkins pipeline and recipes"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress."
        )
    
    try:
        print(f"Yocto build prediction request from {user_id}")
        print(f"Target image: {request.target_image}")
        
        # Combine all build inputs
        build_input = f"""
=== JENKINS PIPELINE ===
{request.jenkins_pipeline}

=== YOCTO RECIPES ===  
{request.yocto_recipes}

=== BUILD CONFIGURATION ===
{request.build_config}

=== TARGET IMAGE ===
{request.target_image}
"""
        
        if not any([request.jenkins_pipeline.strip(), request.yocto_recipes.strip(), request.build_config.strip()]):
            raise HTTPException(status_code=400, detail="No build configuration provided")
        
        # Use existing RAG analysis function
        prediction_result = await get_rag_response_async(build_input, user_id)
        sanitized_response = sanitize_response_content(prediction_result)
        
        return {
            "target_image": request.target_image,
            "prediction": sanitized_response,
            "status": "completed",
            "analyzed_by": "yocto-build-predictor",
            "timestamp": int(time.time())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in Yocto build prediction for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ========= MANAGEMENT ENDPOINTS =========
@app.post("/reindex")
async def reindex_knowledge_base():
    """Manually trigger reindexing of knowledge base"""
    try:
        print("MANUAL REINDEX - DELETING OLD STORE...")
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        
        await initialize_rag_system_async()
        return {
            "status": "success", 
            "message": "Reindexed with fresh MinIO data",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG API Service"}


@app.get("/users/active")
async def get_active_users():
    """Get list of users with active requests"""
    active_users = [user_id for user_id, lock in user_locks.items() if lock.locked()]
    return {"active_users": active_users, "total_active": len(active_users)}


@app.get("/watcher/status")
async def get_watcher_status():
    """Get current status of file watcher"""
    global file_observer, last_change_time
    
    return {
        "enabled": WATCH_ENABLED,
        "running": file_observer is not None and file_observer.is_alive(),
        "watch_path": os.path.abspath(os.path.expanduser(MINIO_DATA_PATH)),
        "target_bucket": MINIO_BUCKET,
        "debounce_seconds": DEBOUNCE_SECONDS,
        "last_change_time": last_change_time,
        "refresh_in_progress": refresh_lock.locked(),
        "force_rebuild_on_startup": FORCE_REBUILD_ON_STARTUP,
        "check_minio_on_every_query": CHECK_MINIO_ON_EVERY_QUERY
    }


@app.post("/watcher/restart")
async def restart_watcher():
    """Restart the file watcher"""
    try:
        stop_file_watcher()
        loop = asyncio.get_event_loop()
        new_observer = setup_file_watcher(loop)
        
        if new_observer:
            return {"status": "success", "message": "File watcher restarted successfully"}
        else:
            return {"status": "error", "message": "Failed to start file watcher"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorstore/status")
async def vectorstore_status():
    """Get current vectorstore status and statistics"""
    try:
        if vectorstore is None:
            return {"status": "not_initialized"}
        
        total_docs = len(vectorstore.docstore._dict) if vectorstore.docstore else 0
        file_count = len(file_tracker.file_metadata)
        
        return {
            "status": "active",
            "total_documents": total_docs,
            "tracked_files": file_count,
            "vectorstore_path": VECTOR_STORE_PATH,
            "last_files": list(file_tracker.file_metadata.keys())[:10]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/vectorstore/force-sync")
async def force_vectorstore_sync():
    """Force synchronization with MinIO without rebuilding"""
    try:
        await run_in_threadpool(perform_incremental_update)
        return {"status": "success", "message": "Vectorstore synchronized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========= MODEL INFO ENDPOINTS =========
@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "data": [
            {
                "id": "codellama:7b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-rag-system"
            },
            {
                "id": "codellama-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-rag-system"
            }
        ]
    }


@app.post("/v1/api/show")
async def show_model_info(request: dict = None):
    """Handle Continue.dev's model info requests"""
    try:
        if request is None:
            request = {}
        model_name = request.get("name", "codellama:7b")
        
        return {
            "license": "Apache 2.0",
            "modelfile": f"# Modelfile for {model_name}",
            "parameters": "temperature 0.4\nmax_tokens 2000",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": None,
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/api/show/{model_name}")
async def show_model_info_get(model_name: str):
    """Handle GET requests for model info"""
    return await show_model_info({"name": model_name})


@app.get("/api/tags")
async def get_ollama_tags():
    """Ollama-compatible endpoint for listing models"""
    return {
        "models": [
            {
                "name": "codellama:7b",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 3825819519,
                "digest": "sha256:example",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            },
            {
                "name": "codellama-rag", 
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 3825819519,
                "digest": "sha256:example-rag",
                "details": {
                    "format": "gguf",
                    "family": "llama", 
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }


@app.post("/api/show")
async def show_model_ollama_style(request: dict = None):
    """Ollama-compatible show model endpoint"""
    if request is None:
        request = {}
    
    model_name = request.get("name", "codellama:7b")
    
    return {
        "license": "Apache 2.0",
        "modelfile": f"# Modelfile for {model_name}\nFROM ./model\nTEMPLATE \"{{ .System }}{{ .Prompt }}\"",
        "parameters": "temperature 0.4\nmax_tokens 2000",
        "template": "{{ .System }}{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf", 
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        },
        "model_info": {
            "general.architecture": "llama",
            "general.file_type": 10,
            "general.parameter_count": 6738415616,
            "general.quantization_version": 2
        }
    }


@app.get("/api/version")
async def get_version():
    """Ollama version endpoint"""
    return {"version": "0.1.0"}


# ========= DEBUG ENDPOINTS =========
@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check paths and file detection"""
    watch_path = os.path.abspath(os.path.expanduser(MINIO_DATA_PATH))
    
    debug_info = {
        "config": {
            "minio_data_path": MINIO_DATA_PATH,
            "resolved_watch_path": watch_path,
            "vector_store_path": VECTOR_STORE_PATH,
            "minio_bucket": MINIO_BUCKET,
            "force_rebuild_on_startup": FORCE_REBUILD_ON_STARTUP,
            "check_minio_on_every_query": CHECK_MINIO_ON_EVERY_QUERY
        },
        "path_exists": os.path.exists(watch_path),
        "vector_store_exists": os.path.exists(VECTOR_STORE_PATH),
        "build_lock_exists": os.path.exists(BUILD_LOCK_FILE)
    }
    
    try:
        minio_snapshot = get_minio_file_snapshot()
        debug_info["minio_files"] = len(minio_snapshot)
        debug_info["minio_sample"] = list(minio_snapshot.keys())[:5]
    except Exception as e:
        debug_info["minio_error"] = str(e)
    
    return debug_info

@app.get("/debug/context-analysis")
async def debug_context_analysis():
    """Debug what context is being sent to LLM"""
    try:
        # Get sample documents
        sample_docs = vectorstore.similarity_search("yocto jenkins pipeline", k=50)
        
        # Separate documents
        confluence_docs = []
        log_docs = []
        
        for doc in sample_docs:
            source = doc.metadata.get('source', '')
            if any(keyword in source.lower() for keyword in ['static_knowledge', 'yocto', 'build', 'guideline', 'documentation']):
                confluence_docs.append(doc)
            elif any(keyword in source.lower() for keyword in ['dynamic_knowledge', 'log', 'jenkins_log', 'build_logs']):
                log_docs.append(doc)
            else:
                confluence_docs.append(doc)
        
        # Build contexts
        guidelines_context = build_confluence_context(confluence_docs[:10])
        logs_context = build_log_context(log_docs[:5])
        
        # Show actual context being sent to LLM
        return {
            "confluence_docs_count": len(confluence_docs),
            "log_docs_count": len(log_docs),
            "guidelines_context_length": len(guidelines_context),
            "logs_context_length": len(logs_context),
            "guidelines_context_preview": guidelines_context[:1000],
            "logs_context_preview": logs_context[:500],
            "full_guidelines_context": guidelines_context,
            "full_logs_context": logs_context
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/minio")
async def debug_minio():
    """Debug MinIO contents and vector store status"""
    try:
        current_snapshot = get_minio_file_snapshot()
        
        # Get build metadata
        metadata_file = os.path.join(VECTOR_STORE_PATH, "build_metadata.json")
        build_info = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                build_info = json.load(f)
        
        return {
            "current_minio_files": current_snapshot,
            "minio_file_count": len(current_snapshot),
            "vector_store_exists": os.path.exists(VECTOR_STORE_PATH),
            "build_metadata": build_info,
            "last_build_time": build_info.get('build_datetime') if build_info else None,
            "total_chunks_in_store": build_info.get('total_chunks') if build_info else None,
            "files_processed_last_build": build_info.get('files_processed') if build_info else None
        }
    except Exception as e:
        return {"error": str(e)}


def setup_signal_handlers():
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}. Shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    
    # Install required package s first
    print("Make sure you have installed the required packages:")
    print("pip install -U langchain-ollama")
    
    uvicorn.run(
        "feedbacks:app",
        host="0.0.0.0", 
        port=8000,
        workers=1,
        reload=False
    )
