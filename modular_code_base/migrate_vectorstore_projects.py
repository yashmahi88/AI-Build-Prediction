"""
One-time migration to add project_id to existing vectorstore documents
Run: python migrate_vectorstore_projects.py
"""
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from app.utils.project_identifier import extract_project_id_from_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_vectorstore(vectorstore_path='./vectorstore'):
    """Add project_id to all existing documents"""
    
    logger.info(f"Loading vectorstore from {vectorstore_path}")
    
    # Load embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Load vectorstore
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    logger.info(f"Loaded {vectorstore.index.ntotal} vectors")
    
    # Access docstore
    docstore = vectorstore.docstore._dict
    
    updated = 0
    
    for doc_id, doc in docstore.items():
        # Skip if already has project_id
        if 'project_id' in doc.metadata:
            continue
        
        # Try to infer project from existing metadata
        job_name = doc.metadata.get('job_name', '')
        source = doc.metadata.get('source', '')
        
        if job_name:
            project_id = extract_project_id_from_job(job_name)
        elif 'jenkins' in source.lower():
            # Try to extract job from source path
            parts = source.split('/')
            if len(parts) >= 2:
                job_name = parts[1]
                project_id = extract_project_id_from_job(job_name)
            else:
                project_id = 'default'
        else:
            project_id = 'default'
        
        # Add project_id
        doc.metadata['project_id'] = project_id
        doc.metadata['migrated'] = True
        updated += 1
        
        if updated % 100 == 0:
            logger.info(f"Updated {updated} documents...")
    
    logger.info(f"Updated {updated} total documents")
    
    # Backup
    backup_path = f"{vectorstore_path}.backup"
    logger.info(f"Creating backup at {backup_path}")
    import shutil
    if os.path.exists(vectorstore_path):
        shutil.copytree(vectorstore_path, backup_path, dirs_exist_ok=True)
    
    # Save
    logger.info(f"Saving to {vectorstore_path}")
    vectorstore.save_local(vectorstore_path)
    
    logger.info("✅ Migration complete!")
    
    return updated

if __name__ == "__main__":
    migrate_vectorstore()
