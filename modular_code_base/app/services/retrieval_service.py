<<<<<<< Updated upstream
"""Document retrieval from vectorstore"""
import logging
from typing import List
from langchain.schema import Document

logger = logging.getLogger(__name__)
=======
"""Document retrieval from vectorstore"""  # Module docstring describing this file handles semantic search over vectorstore to find relevant documents
import logging  # Standard Python logging library for tracking retrieval operations
from typing import List  # Type hint for function return types: List for arrays
from langchain_core.documents import Document  # LangChain's Document class representing text chunks with metadata
>>>>>>> Stashed changes


logger = logging.getLogger(__name__)  # Create logger instance for this module to output retrieval-related logs



class RetrievalService:  # Service class that provides document retrieval functionality from FAISS vectorstore
    """Retrieve relevant documents from vectorstore"""  # Docstring explaining this class performs semantic search to find relevant documents
    
    def __init__(self, vectorstore_service):  # Constructor that initializes the retrieval service with a vectorstore service instance
        self.vectorstore_service = vectorstore_service  # Store reference to vectorstore service (provides search functionality)
        logger.info(f"RetrievalService initialized with vectorstore")  # Log successful initialization
    
    def retrieve_relevant_documents(self, query: str, k: int = 20) -> List[Document]:  # Main method to perform semantic search and retrieve top-k most relevant documents
        """Retrieve documents relevant to query"""  # Docstring explaining this method searches vectorstore for documents similar to the query
        
        #  CHECK IF VECTORSTORE IS LOADED
        if not self.vectorstore_service.is_loaded():  # Check if vectorstore index is available in memory (FAISS index loaded successfully)
            logger.warning(" Vectorstore not loaded")  # Log warning that vectorstore is unavailable 
            return []  # Return empty list since we can't search without a loaded vectorstore
        
        try:  # Wrap retrieval in try-except to handle search errors gracefully
            logger.info(f" Retrieving {k} documents for query...")  # Log that we're starting retrieval with specified k value 
            docs = self.vectorstore_service.search(query, k=k)  # Perform vector similarity search to find k most similar documents to the query (uses FAISS cosine similarity or L2 distance)
            logger.info(f"✅ Retrieved {len(docs)} documents")  # Log successful retrieval with actual number of documents returned (may be less than k if vectorstore has fewer documents)
            
            # ✅ ENSURE METADATA ON EACH DOC
            for i, doc in enumerate(docs):  # Loop through each retrieved document with index (enumerate provides index counter)
                if not hasattr(doc, 'metadata'):  # Check if document object has metadata attribute (defensive check for malformed documents)
                    doc.metadata = {}  # Initialize empty metadata dictionary if missing (ensures all documents have metadata field)
                
                # Add source metadata if missing
                if 'source' not in doc.metadata:  # Check if metadata has 'source' field (indicates where document came from)
                    doc.metadata['source'] = f'Vectorstore-Doc-{i}'  # Generate default source identifier using document index (e.g., "Vectorstore-Doc-0")
                    doc.metadata['confluence_url'] = f''  # Add empty confluence_url field (prevents KeyError in downstream code that expects this field)
            
            return docs  # Return list of retrieved documents with ensured metadata
        
<<<<<<< Updated upstream
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []
=======
        except Exception as e:  # Catch any errors during retrieval (vectorstore errors, search failures, etc.)
            logger.error(f"❌ Error retrieving documents: {e}")  # Log error with details 
            return []  # Return empty list on error (fail gracefully so analysis can continue with workspace rules)
>>>>>>> Stashed changes
