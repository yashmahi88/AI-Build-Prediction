import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vectorstore_service import VectorStoreService

# Path to the text file you want to add
file_path = "/home/azureuser/rag-system/updatetest.txt"

# Read the file content
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split content into chunks similar to pipeline
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(content)

# Create Document objects to add to vectorstore
documents = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": "manual_upload",
            "file_name": os.path.basename(file_path),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
    )
    documents.append(doc)

# Initialize your vectorstore service pointing to the correct path
vectorstore_path = "/home/azureuser/rag-system/modular_code_base/vectorstore"
service = VectorStoreService(vector_store_path=vectorstore_path)

# Add documents (this will create or update the existing vectorstore)
success = service.add_documents(documents)
if success:
    print(f"Added {len(documents)} chunks from {file_path} to vectorstore")
else:
    print("Failed to add documents to vectorstore")
