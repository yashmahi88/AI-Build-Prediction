"""Vector store management service"""
import os
import logging
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages FAISS vector store operations"""
    
    def __init__(self, vector_store_path: str = "./vectorstore"):
        """Initialize VectorStoreService"""
        self.vector_store_path = vector_store_path
        self.vectorstore = None
        self.embeddings = None
        
        logger.info(f"Initializing VectorStoreService with path: {vector_store_path}")
        
        try:
            self.embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434",
                model="nomic-embed-text"
            )
            logger.info("✅ Embeddings initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            self.embeddings = None
        
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Automatically load vectorstore if it exists"""
        if self._exists():
            logger.info(f"Vectorstore found at: {self.vector_store_path}")
            try:
                self.load()
                logger.info("✅ Vectorstore loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load vectorstore: {e}")
                self.vectorstore = None
        else:
            logger.warning(f"Vectorstore not found at: {self.vector_store_path}")
            logger.warning("First documents added will create the vectorstore")
    
    def _exists(self) -> bool:
        """Check if vector store exists"""
        exists = os.path.exists(self.vector_store_path)
        logger.debug(f"Vectorstore exists check: {exists}")
        return exists
    
    def load(self) -> Optional[FAISS]:
        """Load existing vector store from disk"""
        if self.embeddings is None:
            logger.error("❌ Embeddings not initialized - cannot load vectorstore")
            return None
        
        try:
            logger.info(f"Loading vectorstore from: {self.vector_store_path}")
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ Loaded vector store successfully")
            return self.vectorstore
        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {e}")
            self.vectorstore = None
            return None
    
    def load_or_build(self, force_rebuild: bool = False):
        """Load existing vectorstore or build new one - ADDED METHOD"""
        if force_rebuild:
            logger.info("Force rebuild requested - clearing vectorstore")
            self.vectorstore = None
            
            # Rebuild from MinIO
            try:
                from app.services.minio_service import MinIOService
                from app.services.document_processor import DocumentProcessor
                
                minio_service = MinIOService()
                doc_processor = DocumentProcessor()
                
                # Get files from MinIO
                files = minio_service.list_files()
                documents = []
                
                for file in files:
                    content = minio_service.get_file_content(file['Key'])
                    if content:
                        docs = doc_processor.process_text(
                            content,
                            metadata={'source': file['Key']}
                        )
                        documents.extend(docs)
                
                if documents:
                    logger.info(f"Building vectorstore with {len(documents)} documents")
                    self.build(documents)
                else:
                    logger.warning("No documents to build vectorstore")
                    
            except Exception as e:
                logger.error(f"❌ Error rebuilding vectorstore: {e}")
        else:
            # Normal load
            if not self._exists():
                logger.warning("Vectorstore doesn't exist yet")
            elif self.vectorstore is None:
                self._load_vectorstore()
    
    def build(self, documents: List[Document]) -> Optional[FAISS]:
        """Build new vector store from documents"""
        if not documents:
            logger.warning("No documents provided to build vectorstore")
            return None
        
        if self.embeddings is None:
            logger.error("❌ Embeddings not initialized")
            return None
        
        try:
            logger.info(f"Building new vectorstore with {len(documents)} documents")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.save()
            logger.info(f"✅ Built and saved vector store with {len(documents)} documents")
            return self.vectorstore
        except Exception as e:
            logger.error(f"❌ Error building vectorstore: {e}")
            self.vectorstore = None
            return None
    
    def save(self):
        """Persist vector store to disk"""
        if self.vectorstore is None:
            logger.warning("No vectorstore to save")
            return
        
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vectorstore.save_local(self.vector_store_path)
            logger.info(f"✅ Vector store saved to: {self.vector_store_path}")
        except Exception as e:
            logger.error(f"❌ Error saving vectorstore: {e}")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to existing vector store or create new one"""
        if not documents:
            logger.warning("No documents to add")
            return False
        
        if self.embeddings is None:
            logger.error("❌ Embeddings not initialized")
            return False
        
        try:
            if self.vectorstore is None:
                logger.info(f"Creating new vectorstore with {len(documents)} documents")
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                logger.info(f"Adding {len(documents)} documents to existing vectorstore")
                self.vectorstore.add_documents(documents)
            
            self.save()
            logger.info(f"✅ Documents added and vectorstore saved")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False
    
    def search(self, query: str, k: int = 20) -> List[Document]:
        """Search vectorstore by similarity"""
        if self.vectorstore is None:
            logger.warning("Vectorstore not loaded - cannot search")
            return []
        
        try:
            logger.debug(f"Searching vectorstore for: {query[:50]}...")
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"✅ Found {len(results)} matching documents")
            return results
        except Exception as e:
            logger.error(f"❌ Error searching vectorstore: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """Check if vectorstore is loaded"""
        return self.vectorstore is not None
