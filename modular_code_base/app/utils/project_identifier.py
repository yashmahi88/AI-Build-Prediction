"""
Project identification utility
Extracts project_id from various sources (job names, workspace paths, etc.)
"""
import logging
import re

logger = logging.getLogger(__name__)


def extract_project_id_from_job(job_name: str) -> str:
    """
    Extract project_id from Jenkins job name
    
    Examples:
    - "Yocto-ProjectA-Build" → "projecta"
    - "ProjectB-Integration-Pipeline" → "projectb"
    - "customer-xyz-build" → "customer-xyz"
    - "Yocto-Build-Pipeline" → "default"
    """
    if not job_name:
        return 'default'
    
    # Remove common suffixes
    job_name_lower = job_name.lower()
    for suffix in ['-build', '-pipeline', '-integration', '-test', '-deploy']:
        job_name_lower = job_name_lower.replace(suffix, '')
    
    # Extract parts
    parts = job_name_lower.split('-')
    
    # Skip generic prefixes
    skip_prefixes = ['yocto', 'jenkins', 'ci', 'cd']
    project_parts = [p for p in parts if p not in skip_prefixes and len(p) > 1]
    
    if project_parts:
        # Return first meaningful part
        project_id = project_parts[0]
        logger.debug(f"Extracted project_id '{project_id}' from job '{job_name}'")
        return project_id
    
    logger.debug(f"No specific project identified from job '{job_name}', using 'default'")
    return 'default'


def extract_project_id_from_workspace(workspace_path: str) -> str:
    """
    Extract project_id from workspace path
    
    Examples:
    - "/var/jenkins_home/workspace/ProjectA-Build" → "projecta"
    - "/yocto-builds/customer-xyz" → "customer-xyz"
    """
    if not workspace_path:
        return 'default'
    
    # Get last directory component
    parts = workspace_path.rstrip('/').split('/')
    workspace_name = parts[-1] if parts else ''
    
    # Extract project from workspace name
    return extract_project_id_from_job(workspace_name)


def normalize_project_id(project_id: str) -> str:
    """Normalize project_id to lowercase alphanumeric with hyphens"""
    if not project_id:
        return 'default'
    
    # Convert to lowercase, replace underscores with hyphens
    normalized = project_id.lower().replace('_', '-')
    
    # Remove any non-alphanumeric except hyphens
    normalized = re.sub(r'[^a-z0-9-]', '', normalized)
    
    # Remove consecutive hyphens
    normalized = re.sub(r'-+', '-', normalized)
    
    # Strip leading/trailing hyphens
    normalized = normalized.strip('-')
    
    return normalized if normalized else 'default'
