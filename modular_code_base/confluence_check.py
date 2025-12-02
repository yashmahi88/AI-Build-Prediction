# add_confluence_urls.py
import sys
sys.path.insert(0, '/home/azureuser/rag-system/modular_code_base')

from app.services.vectorstore_service import VectorStoreService

# Load service
service = VectorStoreService("./vectorstore")

if not service.is_loaded():
    print("Failed to load vectorstore")
    sys.exit(1)

CONFLUENCE_BASE = "https://forexzig-team-gfg1ft1k.atlassian.net/wiki"

fixed_count = 0

for doc_id, doc in service.vectorstore.docstore._dict.items():
    source = doc.metadata.get('source', '')
    
    if 'confluence' in source.lower():
        current_url = doc.metadata.get('confluence_url', '')
        
        if not current_url or current_url == 'MISSING':
            space = doc.metadata.get('space', 'OH')
            page_id = doc.metadata.get('page_id', '')
            
            if page_id and space:
                confluence_url = f"{CONFLUENCE_BASE}/spaces/{space}/pages/{page_id}"
                doc.metadata['confluence_url'] = confluence_url
                fixed_count += 1

print(f"Fixed {fixed_count} Confluence documents")

if fixed_count > 0:
    print("Saving changes...")
    service.save()
    print("Done! URLs added:")
    print(f"  1. {CONFLUENCE_BASE}/spaces/OH/pages/131210")
    print(f"  2. {CONFLUENCE_BASE}/spaces/OH/pages/50298881")
    print("\nRestart your app to use updated vectorstore")
