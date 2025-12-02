# test_retrieval.py
import sys
sys.path.insert(0, '/home/azureuser/rag-system/modular_code_base')

from app.services.vectorstore_service import VectorStoreService
from app.services.retrieval_service import RetrievalService

# Load services
vectorstore_service = VectorStoreService("./vectorstore")
retrieval_service = RetrievalService(vectorstore_service)

# Test query (same as your pipeline)
query = """
YOEDISTRO_REPO=https://github.com/YoeDistro/yoe-distro.git
TARGET_MACHINE=rpi4-64
IMAGE=yoe-simple-image
DISTRO=yoe-distro
"""

print("Testing retrieval with Yoe-Distro query...\n")
docs = retrieval_service.retrieve_relevant_documents(query, k=50)

print(f"Retrieved {len(docs)} documents\n")
print("="*60)

for i, doc in enumerate(docs[:10], 1):
    source = doc.metadata.get('source', 'N/A')
    title = doc.metadata.get('title', 'N/A')
    page_id = doc.metadata.get('page_id', 'N/A')
    
    # Check if it's Confluence
    is_confluence = 'confluence' in source.lower()
    
    if is_confluence:
        print(f"\n{i}. [CONFLUENCE] {title}")
        print(f"   Page ID: {page_id}")
        print(f"   Source: {source}")
        print(f"   Content preview: {doc.page_content[:100]}...")
    else:
        print(f"\n{i}. [OTHER] {source[:60]}")
