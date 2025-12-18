"""Vector store management service"""  # Module docstring describing this file manages FAISS vectorstore operations (load, save, search, add documents)
import os  # Operating system interface for file and directory operations
import logging  # Standard Python logging library for tracking vectorstore operations
from typing import List, Optional  # Type hints for function signatures: List for arrays, Optional for nullable types


from langchain_community.vectorstores import FAISS  # LangChain's FAISS wrapper for vector similarity search (Facebook AI Similarity Search)
from langchain_ollama import OllamaEmbeddings  # LangChain wrapper for Ollama embeddings (converts text to vectors using local Ollama models)
from langchain_core.documents import Document  # LangChain's Document class representing text chunks with metadata


logger = logging.getLogger(__name__)  # Create logger instance for this module to output vectorstore-related logs



class VectorStoreService:  # Service class that manages FAISS vectorstore lifecycle (initialization, loading, saving, searching)
    """Manages FAISS vector store operations"""  # Docstring explaining this class handles all vectorstore operations
    
    def __init__(self, vector_store_path: str = "./vectorstore"):  # Constructor that initializes vectorstore service with path to FAISS index files
        """Initialize VectorStoreService"""  # Docstring for constructor
        self.vector_store_path = vector_store_path  # Store path where FAISS index files are saved (index.faiss and index.pkl)
        self.vectorstore = None  # Initialize vectorstore as None (will be loaded later)
        self.embeddings = None  # Initialize embeddings model as None (will be created next)
        
        logger.info(f"Initializing VectorStoreService with path: {vector_store_path}")  # Log initialization with configured path
        
        try:  # Wrap embeddings initialization in try-except to handle connection errors
            self.embeddings = OllamaEmbeddings(  # Create Ollama embeddings instance for converting text to vectors
                base_url="http://localhost:11434",  # Ollama server URL (default Ollama port)
                model="nomic-embed-text:latest",  # Embedding model name (nomic-embed-text is optimized for retrieval, alternative: mxbai-embed-large)
                num_ctx=8192,        # Context window size (must match model's context length - 8192 tokens for nomic-embed-text)
                num_thread=4,  # Number of CPU threads to use for embedding generation (parallel processing)
            )
            logger.info("✅ Embeddings initialized")  # Log successful embeddings initialization
        except Exception as e:  # Catch any errors during embeddings initialization (Ollama not running, model not found, etc.)
            logger.error(f"❌ Failed to initialize embeddings: {e}")  # Log error details
            self.embeddings = None  # Set embeddings to None so other methods know it's unavailable
        
        self._load_vectorstore()  # Attempt to load existing vectorstore from disk (if it exists)
    
    def _load_vectorstore(self):  # Private method to automatically load vectorstore from disk if it exists
        """Automatically load vectorstore if it exists"""  # Docstring explaining this method attempts to load existing index on startup
        if self._exists():  # Check if vectorstore files exist on disk (index.faiss and index.pkl)
            logger.info(f"Vectorstore found at: {self.vector_store_path}")  # Log that vectorstore files were found
            try:  # Wrap loading in try-except to handle corrupted index files
                self.load()  # Call load method to read FAISS index into memory
                logger.info("✅ Vectorstore loaded successfully")  # Log successful loading
            except Exception as e:  # Catch any errors during loading (corrupted files, wrong format, etc.)
                logger.error(f"❌ Failed to load vectorstore: {e}")  # Log error details
                self.vectorstore = None  # Set vectorstore to None so searches fail gracefully
        else:  # Vectorstore files don't exist
            logger.warning(f"Vectorstore not found at: {self.vector_store_path}")  # Log warning that vectorstore needs to be built
            logger.warning("First documents added will create the vectorstore")  # Log that vectorstore will be created when documents are added
    
    def _exists(self) -> bool:  # Private method to check if vectorstore files exist on disk
        """Check if vector store exists"""  # Docstring explaining this checks for FAISS index files
        exists = os.path.exists(self.vector_store_path)  # Check if vectorstore directory exists (contains index.faiss and index.pkl files)
        logger.debug(f"Vectorstore exists check: {exists}")  # Log debug message with existence status
        return exists  # Return True if vectorstore directory exists, False otherwise
    
    def load(self) -> Optional[FAISS]:  # Method to load existing FAISS index from disk into memory
        """Load existing vector store from disk"""  # Docstring explaining this reads FAISS index files into memory
        if self.embeddings is None:  # Check if embeddings model is available (required for loading vectorstore)
            logger.error("❌ Embeddings not initialized - cannot load vectorstore")  # Log error that embeddings are missing
            return None  # Return None since we can't load without embeddings
        
        try:  # Wrap loading in try-except to handle file errors
            logger.info(f"Loading vectorstore from: {self.vector_store_path}")  # Log start of loading operation
            self.vectorstore = FAISS.load_local(  # Load FAISS index from disk using LangChain's load_local method
                self.vector_store_path,  # Path to directory containing index.faiss and index.pkl
                self.embeddings,  # Embeddings model instance (must match the model used to create the index)
                allow_dangerous_deserialization=True  # Allow pickle deserialization (required for loading FAISS index metadata, be careful with untrusted sources)
            )
            logger.info(f"✅ Loaded vector store successfully")  # Log successful loading
            return self.vectorstore  # Return loaded FAISS instance
        except Exception as e:  # Catch any errors during loading (file not found, corrupted index, wrong embeddings model, etc.)
            logger.error(f"❌ Error loading vectorstore: {e}")  # Log error details
            self.vectorstore = None  # Set vectorstore to None to indicate loading failed
            return None  # Return None to indicate failure
    
    def load_or_build(self, force_rebuild: bool = False):  # Method to load existing vectorstore or rebuild from sources (MinIO + Yocto docs)
        """Load existing vectorstore or build new one - ADDED METHOD
        
        NOTE: This method rebuilds from your configured sources (MinIO + Yocto docs)
        when force_rebuild is True, but the public name and signature remain unchanged.
        """  # Detailed docstring explaining rebuild functionality
        if force_rebuild:  # Check if caller requested force rebuild (ignore existing index)
            logger.info("Force rebuild requested - clearing vectorstore")  # Log that we're rebuilding from scratch
            self.vectorstore = None  # Clear existing vectorstore reference
            
            try:  # Wrap rebuild in try-except to handle errors during document collection
                # Import your source services here
                from app.services.minio_service import MinIOService  # Import MinIO service for fetching user-uploaded files (lazy import to avoid circular dependencies)
                from app.services.document_processor import DocumentProcessor  # Import document processor for chunking and formatting
                from app.services.yocto_docs_service import scrape_yocto_docs  # Import Yocto documentation scraper
                
                minio_service = MinIOService()  # Create MinIO service instance to access object storage
                doc_processor = DocumentProcessor()  # Create document processor instance for text chunking
                
                documents: List[Document] = []  # Initialize empty list to collect all documents for vectorstore
                
                # 1) Get files from MinIO (if you still use it)
                files = minio_service.list_files()  # List all files in MinIO bucket
                logger.info(f"Rebuild from MinIO: {len(files)} files found")  # Log number of files found in MinIO
                for file in files:  # Loop through each file in MinIO
                    content = minio_service.get_file_content(file["Key"])  # Fetch file content from MinIO by key (file path)
                    if content:  # If file content was successfully retrieved
                        docs = doc_processor.process_text(  # Process and chunk the content into Document objects
                            content,  # Raw text content from file
                            metadata={'source': file["Key"]}  # Add source metadata with file key (for citation tracking)
                        )
                        documents.extend(docs)  # Add all chunks to documents list
                
                # 2) Yocto docs from web
                yocto_docs = scrape_yocto_docs()  # Scrape Yocto documentation from official website (returns list of Document objects)
                if yocto_docs:  # If scraping returned documents
                    logger.info(f"Adding {len(yocto_docs)} Yocto docs during rebuild")  # Log number of Yocto docs being added
                    documents.extend(yocto_docs)  # Add Yocto documentation to documents list
                
                if documents:  # If we collected any documents from any source
                    logger.info(f"Building vectorstore with {len(documents)} documents")  # Log total document count for vectorstore
                    self.build(documents)  # Build new FAISS index from all documents
                else:  # No documents were collected from any source
                    logger.warning("No documents to build vectorstore")  # Log warning that vectorstore is empty
                    
            except Exception as e:  # Catch any errors during rebuild process
                logger.error(f"❌ Error rebuilding vectorstore: {e}")  # Log error details
        else:  # Not a force rebuild, just normal load
            # Normal load
            if not self._exists():  # Check if vectorstore exists
                logger.warning("Vectorstore doesn't exist yet")  # Log that vectorstore needs to be created
            elif self.vectorstore is None:  # Vectorstore exists but isn't loaded yet
                self._load_vectorstore()  # Load existing vectorstore from disk
    
    def build(self, documents: List[Document]) -> Optional[FAISS]:  # Method to create new FAISS index from a list of documents
        """Build new vector store from documents"""  # Docstring explaining this creates fresh vectorstore from documents
        if not documents:  # Check if documents list is empty
            logger.warning("No documents provided to build vectorstore")  # Log warning that there's nothing to build
            return None  # Return None since we can't build without documents
        
        if self.embeddings is None:  # Check if embeddings model is available (required for vectorstore)
            logger.error("❌ Embeddings not initialized")  # Log error that embeddings are missing
            return None  # Return None since we can't build without embeddings
        
        try:  # Wrap building in try-except to handle errors
            logger.info(f"Building new vectorstore with {len(documents)} documents")  # Log start of build operation with document count
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)  # Create new FAISS index from documents (generates embeddings for all document chunks)
            self.save()  # Save newly built vectorstore to disk
            logger.info(f"✅ Built and saved vector store with {len(documents)} documents")  # Log successful build and save
            return self.vectorstore  # Return newly built FAISS instance
        except Exception as e:  # Catch any errors during build (out of memory, embeddings failure, etc.)
            logger.error(f"❌ Error building vectorstore: {e}")  # Log error details
            self.vectorstore = None  # Set vectorstore to None to indicate build failed
            return None  # Return None to indicate failure
    
    def save(self):  # Method to persist vectorstore to disk (writes index.faiss and index.pkl files)
        """Persist vector store to disk"""  # Docstring explaining this saves FAISS index to filesystem
        if self.vectorstore is None:  # Check if there's a vectorstore to save
            logger.warning("No vectorstore to save")  # Log warning that save was called with no vectorstore loaded
            return  # Exit early since there's nothing to save
        
        try:  # Wrap saving in try-except to handle file system errors
            os.makedirs(self.vector_store_path, exist_ok=True)  # Create vectorstore directory if it doesn't exist (exist_ok=True prevents error if directory already exists)
            self.vectorstore.save_local(self.vector_store_path)  # Save FAISS index to disk (creates index.faiss binary file and index.pkl metadata file)
            logger.info(f"✅ Vector store saved to: {self.vector_store_path}")  # Log successful save with path
        except Exception as e:  # Catch any errors during save (permission denied, disk full, etc.)
            logger.error(f"❌ Error saving vectorstore: {e}")  # Log error details
    
    def add_documents(self, documents: List[Document]) -> bool:  # Method to add new documents to existing vectorstore or create new one if none exists
        """Add documents to existing vector store or create new one"""  # Docstring explaining this incrementally adds documents to vectorstore
        if not documents:  # Check if documents list is empty
            logger.warning("No documents to add")  # Log warning that there's nothing to add
            return False  # Return False to indicate operation failed
        
        if self.embeddings is None:  # Check if embeddings model is available
            logger.error("❌ Embeddings not initialized")  # Log error that embeddings are missing
            return False  # Return False since we can't add without embeddings
        
        try:  # Wrap adding in try-except to handle errors
            if self.vectorstore is None:  # Check if vectorstore doesn't exist yet
                logger.info(f"Creating new vectorstore with {len(documents)} documents")  # Log that we're creating new vectorstore (first documents)
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)  # Create new FAISS index from documents
            else:  # Vectorstore already exists
                logger.info(f"Adding {len(documents)} documents to existing vectorstore")  # Log that we're adding to existing index
                self.vectorstore.add_documents(documents)  # Add documents to existing FAISS index (generates embeddings and updates index)
            
            self.save()  # Save updated vectorstore to disk
            logger.info(f"✅ Documents added and vectorstore saved")  # Log successful addition and save
            return True  # Return True to indicate success
        except Exception as e:  # Catch any errors during addition (embeddings failure, FAISS error, etc.)
            logger.error(f"❌ Error adding documents: {e}")  # Log error details
            return False  # Return False to indicate failure
    
    def search(self, query: str, k: int = 20) -> List[Document]:  # Method to perform semantic similarity search on vectorstore
        """Search vectorstore by similarity"""  # Docstring explaining this finds most similar documents to query
        if self.vectorstore is None:  # Check if vectorstore is loaded
            logger.warning("Vectorstore not loaded - cannot search")  # Log warning that search can't be performed
            return []  # Return empty list since we can't search without vectorstore
        
        try:  # Wrap searching in try-except to handle errors
            logger.debug(f"Searching vectorstore for: {query[:50]}...")  # Log start of search with truncated query ([:50] shows first 50 chars)
            results = self.vectorstore.similarity_search(query, k=k)  # Perform FAISS similarity search (converts query to embedding, finds k nearest neighbors using cosine similarity or L2 distance)
            logger.info(f"✅ Found {len(results)} matching documents")  # Log number of results found (may be less than k if vectorstore has fewer documents)
            return results  # Return list of matching Document objects (sorted by similarity score, most similar first)
        except Exception as e:  # Catch any errors during search (vectorstore error, embeddings failure, etc.)
            logger.error(f"❌ Error searching vectorstore: {e}")  # Log error details
            return []  # Return empty list on error (fail gracefully)
    
    def is_loaded(self) -> bool:  # Method to check if vectorstore is currently loaded in memory
        """Check if vectorstore is loaded"""  # Docstring explaining this returns vectorstore availability status
        return self.vectorstore is not None  # Return True if vectorstore is loaded, False if None (simple None check)
