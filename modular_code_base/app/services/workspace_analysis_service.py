"""Yocto workspace analysis service"""
import os
import re
from typing import List, Dict

class WorkspaceAnalysisService:
    """Analyzes actual Yocto workspace files"""
    
    def __init__(self, workspace_paths: List[str] = None):
        self.workspace_paths = workspace_paths or [
            "/var/jenkins_home/workspace/Yocto-Build-Pipeline",
            "/yocto-builds",
            os.getenv("WORKSPACE", "/tmp")
        ]
    
    def _analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze individual Yocto file - NOW READS CONTENT"""
        rules = []
        try:
            # ✅ READ FILE CONTENT (was missing)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()[:10000]  # First 10KB
            
            file_name = os.path.basename(file_path)
            
            # Extract DEPENDS
            depends = re.findall(r'DEPENDS\s*[+=]*\s*["\']([^\'"]+)["\']', content)
            for dep_list in depends:
                for dep in dep_list.split()[:5]:
                    rules.append({
                        'rule_text': f'Recipe {file_name} requires dependency: {dep}',
                        'rule_type': 'YOCTO_DEPENDENCY',
                        'confidence': 0.90,
                        'source': file_name
                    })
            
            # Extract machine config
            machine_match = re.search(r'MACHINE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', content)
            if machine_match:
                rules.append({
                    'rule_text': f'Build targets machine: {machine_match.group(1)}',
                    'rule_type': 'YOCTO_MACHINE',
                    'confidence': 0.95,
                    'source': file_name
                })
            
            # Extract parallel threads
            threads_match = re.search(r'BB_NUMBER_THREADS\s*\??=\s*["\']?(\d+)["\']?', content)
            if threads_match:
                rules.append({
                    'rule_text': f'BitBake configured for {threads_match.group(1)} threads',
                    'rule_type': 'YOCTO_THREADS',
                    'confidence': 0.90,
                    'source': file_name
                })
            
            # Extract inherit classes
            inherits = re.findall(r'inherit\s+([^\n]+)', content)
            for inherit_list in inherits:
                for cls in inherit_list.split()[:3]:
                    rules.append({
                        'rule_text': f'Recipe {file_name} inherits class: {cls}',
                        'rule_type': 'YOCTO_CLASS',
                        'confidence': 0.85,
                        'source': file_name
                    })
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return rules
