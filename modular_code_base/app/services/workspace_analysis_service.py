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
    
    def discover_yocto_workspace(self) -> List[str]:
        """Discover Yocto workspace directories"""
        found_workspaces = []
        
        for path in self.workspace_paths:
            if os.path.exists(path):
                found_workspaces.append(path)
        
        return found_workspaces[:1]  # Only use first workspace
    
    def extract_rules_from_workspace(self, workspace_path: str) -> List[Dict]:
        """Extract rules ONLY from custom layers and build config (NOT poky)"""
        rules = []
        
        if not os.path.exists(workspace_path):
            return rules
        
        # ✅ CRITICAL: Only scan these specific paths
        scan_paths = [
            # Build configuration only
            os.path.join(workspace_path, 'conf/local.conf'),
            os.path.join(workspace_path, 'conf/bblayers.conf'),
            os.path.join(workspace_path, 'build/conf/local.conf'),
            os.path.join(workspace_path, 'build/conf/bblayers.conf'),
        ]
        
        # ✅ Scan custom meta-layers (NOT meta-poky, meta-yocto)
        try:
            for item in os.listdir(workspace_path):
                # Only custom layers (skip standard Poky layers)
                if item.startswith('meta-') and item not in ['meta-poky', 'meta-yocto', 'meta-yocto-bsp']:
                    layer_path = os.path.join(workspace_path, item)
                    if os.path.isdir(layer_path):
                        # Only layer.conf and main recipes
                        scan_paths.append(os.path.join(layer_path, 'conf/layer.conf'))
                        
                        # Scan recipes directory (limit to 5 files)
                        recipes_dir = os.path.join(layer_path, 'recipes')
                        if os.path.exists(recipes_dir):
                            count = 0
                            for root, dirs, files in os.walk(recipes_dir):
                                if count >= 5:
                                    break
                                for file in files:
                                    if file.endswith(('.bb', '.bbappend')) and count < 5:
                                        scan_paths.append(os.path.join(root, file))
                                        count += 1
        except Exception as e:
            pass
        
        # ✅ Extract rules only from identified files
        for file_path in scan_paths:
            if os.path.exists(file_path):
                rules.extend(self._analyze_file(file_path))
        
        return rules
    
    def _analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze individual Yocto file"""
        rules = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()[:10000]  # First 10KB
            
            file_name = os.path.basename(file_path)
            
            # Extract DEPENDS
            depends = re.findall(r'DEPENDS\s*[+=]*\s*["\']([^\'"]+)["\']', content)
            for dep_list in depends[:3]:  # Limit to 3
                for dep in dep_list.split()[:2]:  # Limit to 2 per list
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
            
            # Extract inherit classes (limit)
            inherits = re.findall(r'inherit\s+([^\n]+)', content)
            for inherit_list in inherits[:2]:  # Limit to 2
                for cls in inherit_list.split()[:2]:  # Limit to 2 per line
                    rules.append({
                        'rule_text': f'Recipe {file_name} inherits class: {cls}',
                        'rule_type': 'YOCTO_CLASS',
                        'confidence': 0.85,
                        'source': file_name
                    })
        
        except Exception as e:
            pass
        
        return rules[:20]  # Max 20 rules per file
