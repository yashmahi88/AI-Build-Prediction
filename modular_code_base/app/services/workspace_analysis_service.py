"""Yocto workspace analysis service"""  # Module docstring describing this file scans local Yocto workspace for configuration files and extracts build rules
import os  # Operating system interface for file path operations and directory traversal
import re  # Regular expression library for parsing Yocto configuration syntax
from typing import List, Dict  # Type hints for function signatures: List for arrays, Dict for dictionaries



class WorkspaceAnalysisService:  # Service class that discovers and analyzes Yocto project files in the local workspace
    """Analyzes actual Yocto workspace files"""  # Docstring explaining this class scans workspace for Yocto-specific rules
    
    def __init__(self, workspace_paths: List[str] = None):  # Constructor that initializes service with possible workspace locations
        self.workspace_paths = workspace_paths or [  # Store list of paths to check for Yocto workspaces (use provided paths or default list)
            "/var/jenkins_home/workspace/Yocto-Build-Pipeline",  # Default Jenkins workspace path for CI/CD pipeline
            "/yocto-builds",  # Default local Yocto builds directory
            os.getenv("WORKSPACE", "/tmp")  # Environment variable WORKSPACE if set, otherwise /tmp as fallback (common in CI systems)
        ]
    
    def discover_yocto_workspace(self) -> List[str]:  # Method to find which configured workspace paths actually exist on the filesystem
        """Discover Yocto workspace directories"""  # Docstring explaining this method checks for existing workspace directories
        found_workspaces = []  # Initialize empty list to store discovered workspace paths
        
        for path in self.workspace_paths:  # Loop through each configured workspace path
            if os.path.exists(path):  # Check if directory exists on filesystem
                found_workspaces.append(path)  # Add existing path to found workspaces list
        
        return found_workspaces[:1]  # Return only first workspace found ([:1] limits to single workspace to avoid processing multiple)
    
    def extract_rules_from_workspace(self, workspace_path: str) -> List[Dict]:  # Method to scan workspace and extract Yocto configuration rules from custom layers and build config
        """Extract rules ONLY from custom layers and build config (NOT poky)"""  # Docstring explaining this skips standard Poky layers and focuses on user customizations
        rules = []  # Initialize empty list to collect extracted rules
        
        if not os.path.exists(workspace_path):  # Check if workspace path exists
            return rules  # Return empty list if workspace doesn't exist (can't extract rules from non-existent directory)
        
        # ✅ CRITICAL: Only scan these specific paths
        scan_paths = [  # Initialize list with critical configuration files to analyze
            # Build configuration only
            os.path.join(workspace_path, 'conf/local.conf'),  # Main build configuration file (defines MACHINE, parallel settings, paths, etc.)
            os.path.join(workspace_path, 'conf/bblayers.conf'),  # Layer configuration file (lists which meta-layers are enabled)
            os.path.join(workspace_path, 'build/conf/local.conf'),  # Alternative location for local.conf (in build subdirectory)
            os.path.join(workspace_path, 'build/conf/bblayers.conf'),  # Alternative location for bblayers.conf (in build subdirectory)
        ]
        
        # ✅ Scan custom meta-layers (NOT meta-poky, meta-yocto)
        try:  # Wrap layer scanning in try-except to handle permission errors or missing directories
            for item in os.listdir(workspace_path):  # List all items in workspace directory
                # Only custom layers (skip standard Poky layers)
                if item.startswith('meta-') and item not in ['meta-poky', 'meta-yocto', 'meta-yocto-bsp']:  # Check if item is a custom meta-layer (exclude standard Poky layers that contain thousands of recipes)
                    layer_path = os.path.join(workspace_path, item)  # Build full path to layer directory
                    if os.path.isdir(layer_path):  # Verify it's actually a directory
                        # Only layer.conf and main recipes
                        scan_paths.append(os.path.join(layer_path, 'conf/layer.conf'))  # Add layer.conf (defines layer priority, dependencies, etc.)
                        
                        # Scan recipes directory (limit to 5 files)
                        recipes_dir = os.path.join(layer_path, 'recipes')  # Build path to recipes directory in layer
                        if os.path.exists(recipes_dir):  # Check if recipes directory exists
                            count = 0  # Counter to limit number of recipe files scanned
                            for root, dirs, files in os.walk(recipes_dir):  # Walk recipes directory tree (root=current dir, dirs=subdirs, files=files in current dir)
                                if count >= 5:  # Check if we've already scanned 5 recipe files
                                    break  # Stop scanning to avoid processing too many files
                                for file in files:  # Loop through files in current directory
                                    if file.endswith(('.bb', '.bbappend')) and count < 5:  # Check if file is a recipe (.bb) or recipe append (.bbappend) and we haven't hit limit
                                        scan_paths.append(os.path.join(root, file))  # Add recipe file path to scan list
                                        count += 1  # Increment counter
        except Exception as e:  # Catch any errors during layer scanning (permission denied, etc.)
            pass  # Silently ignore errors and continue with whatever scan_paths we collected
        
        # ✅ Extract rules only from identified files
        for file_path in scan_paths:  # Loop through all collected file paths
            if os.path.exists(file_path):  # Check if file actually exists (some paths may not exist)
                rules.extend(self._analyze_file(file_path))  # Analyze file and add extracted rules to master list
        
        return rules  # Return complete list of extracted rules from workspace
    
    def _analyze_file(self, file_path: str) -> List[Dict]:  # Private method to parse individual Yocto file and extract configuration rules
        """Analyze individual Yocto file"""  # Docstring explaining this method reads and parses a single Yocto configuration or recipe file
        rules = []  # Initialize empty list to store rules extracted from this file
        try:  # Wrap file reading in try-except to handle read errors
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # Open file for reading with UTF-8 encoding (errors='ignore' skips invalid characters instead of failing)
                content = f.read()[:10000]  # Read first 10KB of file ([:10000] limits size to avoid reading huge files into memory)
            
            file_name = os.path.basename(file_path)  # Extract just the filename from full path (for rule metadata)
            
            # Extract DEPENDS
            depends = re.findall(r'DEPENDS\s*[+=]*\s*["\']([^\'"]+)["\']', content)  # Regex to find DEPENDS variable assignments (matches DEPENDS = "dep1 dep2" or DEPENDS += "dep3")
            for dep_list in depends[:3]:  # Loop through first 3 DEPENDS declarations ([:3] limits processing)
                for dep in dep_list.split()[:2]:  # Split dependency list by whitespace and take first 2 dependencies ([:2] limits per declaration)
                    rules.append({  # Add dependency rule
                        'rule_text': f'Recipe {file_name} requires dependency: {dep}',  # Human-readable rule text with filename and dependency name
                        'rule_type': 'YOCTO_DEPENDENCY',  # Categorize as dependency rule
                        'confidence': 0.90,  # Assign 90% confidence (high because DEPENDS syntax is very specific)
                        'source': file_name  # Store source filename for traceability
                    })
            
            # Extract machine config
            machine_match = re.search(r'MACHINE\s*\??=\s*["\']?([^\"\'\n]+)["\']?', content)  # Regex to find MACHINE variable assignment (matches MACHINE = "raspberrypi4" or MACHINE ?= "qemux86")
            if machine_match:  # If MACHINE variable was found
                rules.append({  # Add machine rule
                    'rule_text': f'Build targets machine: {machine_match.group(1)}',  # Extract machine name from first capture group
                    'rule_type': 'YOCTO_MACHINE',  # Categorize as machine configuration rule
                    'confidence': 0.95,  # Assign 95% confidence (MACHINE is critical and syntax is unambiguous)
                    'source': file_name  # Store source filename
                })
            
            # Extract parallel threads
            threads_match = re.search(r'BB_NUMBER_THREADS\s*\??=\s*["\']?(\d+)["\']?', content)  # Regex to find BB_NUMBER_THREADS variable (controls parallel BitBake task execution)
            if threads_match:  # If BB_NUMBER_THREADS was found
                rules.append({  # Add threads rule
                    'rule_text': f'BitBake configured for {threads_match.group(1)} threads',  # Extract thread count from first capture group
                    'rule_type': 'YOCTO_THREADS',  # Categorize as threading configuration rule
                    'confidence': 0.90,  # Assign 90% confidence (threads setting is explicit)
                    'source': file_name  # Store source filename
                })
            
            # Extract inherit classes (limit)
            inherits = re.findall(r'inherit\s+([^\n]+)', content)  # Regex to find inherit statements (recipes inherit classes for common functionality like systemd, cmake, etc.)
            for inherit_list in inherits[:2]:  # Loop through first 2 inherit statements ([:2] limits processing)
                for cls in inherit_list.split()[:2]:  # Split class list by whitespace and take first 2 classes ([:2] limits per statement)
                    rules.append({  # Add inherit rule
                        'rule_text': f'Recipe {file_name} inherits class: {cls}',  # Human-readable rule with filename and class name
                        'rule_type': 'YOCTO_CLASS',  # Categorize as class inheritance rule
                        'confidence': 0.85,  # Assign 85% confidence (inherit syntax is clear but classes are less critical than MACHINE)
                        'source': file_name  # Store source filename
                    })
        
        except Exception as e:  # Catch any errors during file reading or parsing (permission denied, encoding errors, regex failures)
            pass  # Silently ignore errors and return whatever rules we extracted (defensive programming)
        
        return rules[:20]  # Return first 20 rules from this file ([:20] limits output to prevent overwhelming analysis with too many rules from one file)
