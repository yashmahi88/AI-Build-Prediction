# check_vectordb.py
import sys
sys.path.insert(0, '/home/azureuser/rag-system/modular_code_base')

from app.services.vectorstore_service import VectorStoreService

def check_sources():
    # Fix: Use positional argument, not keyword
    service = VectorStoreService("./vectorstore")
    service.load()
    
    if not service.vectorstore:
        print("❌ Vectorstore not loaded")
        return
    
    sources = {}
    
    for doc_id, doc in service.vectorstore.docstore._dict.items():
        source = doc.metadata.get('source', 'NO_SOURCE')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\n📊 Total Documents: {len(service.vectorstore.docstore._dict)}")
    print(f"📁 Unique Sources: {len(sources)}\n")
    
    # Show top 20 sources
    sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 20 Sources:")
    print("-" * 80)
    for source, count in sorted_sources[:20]:
        print(f"{count:>5} docs | {source[:70]}")
    
    # Categorize
    jenkins = sum(1 for s in sources if 'jenkins' in s.lower())
    confluence = sum(1 for s in sources if 'confluence' in s.lower())
    workspace = sum(1 for s in sources if 'workspace' in s.lower())
    minio = sum(1 for s in sources if 'minio' in s.lower())
    
    print("\n" + "=" * 80)
    print(f"Jenkins sources:    {jenkins}")
    print(f"Confluence sources: {confluence}")
    print(f"Workspace sources:  {workspace}")
    print(f"MinIO sources:      {minio}")
    print(f"Other sources:      {len(sources) - jenkins - confluence - workspace - minio}")

if __name__ == "__main__":
    check_sources()
