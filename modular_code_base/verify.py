import sys
sys.path.insert(0, '/home/azureuser/rag-system/modular_code_base')

from app.services.vectorstore_service import VectorStoreService

service = VectorStoreService(vector_store_path='vectorstore')

# Query for unique content in 
query = "Configure Build Environment"

results = service.search(query, k=5)

print(f"Found {len(results)} results for query: '{query}'\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i} snippet: {doc.page_content[:200]} ...")
    print(f"Metadata: {doc.metadata}\n")
