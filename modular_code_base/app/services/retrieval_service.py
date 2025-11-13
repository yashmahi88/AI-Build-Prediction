"""Document retrieval from vectorstore"""
import logging
from typing import List
from langchain.schema import Document

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieve relevant documents from vectorstore"""
    
    def __init__(self, vectorstore_service):
        self.vectorstore_service = vectorstore_service
        logger.info(f"RetrievalService initialized with vectorstore")
    
    def retrieve_relevant_documents(self, query: str, k: int = 20) -> List[Document]:
        """Retrieve documents relevant to query"""
        
        #  CHECK IF VECTORSTORE IS LOADED
        if not self.vectorstore_service.is_loaded():
            logger.warning(" Vectorstore not loaded")
            return []
        
        try:
            logger.info(f" Retrieving {k} documents for query...")
            docs = self.vectorstore_service.search(query, k=k)
            logger.info(f"✅ Retrieved {len(docs)} documents")
            
            # ✅ ENSURE METADATA ON EACH DOC
            for i, doc in enumerate(docs):
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                
                # Add source metadata if missing
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = f'Vectorstore-Doc-{i}'
                    doc.metadata['confluence_url'] = f''
            
            return docs
        
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []
