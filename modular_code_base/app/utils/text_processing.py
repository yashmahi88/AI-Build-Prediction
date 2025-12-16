<<<<<<< Updated upstream
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
=======
from langchain_text_splitters import RecursiveCharacterTextSplitter  # LangChain's text splitter that recursively splits text by separators (newlines, periods, spaces) to create semantically meaningful chunks
from typing import List  # Type hint for list types (imported but not used in this code)
>>>>>>> Stashed changes


def create_confluence_splitter() -> RecursiveCharacterTextSplitter:  # Factory function to create text splitter optimized for documentation (Confluence pages, manuals, guides)
    """Create splitter for documentation (larger chunks)"""  # Docstring explaining this creates splitter with settings appropriate for documentation content
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Create splitter with 500 character chunks and 50 character overlap (overlap preserves context across chunk boundaries and improves retrieval quality)


def create_log_splitter() -> RecursiveCharacterTextSplitter:  # Factory function to create text splitter for log files (Jenkins build logs, console output)
    """Create splitter for logs (smaller chunks)"""  # Docstring explaining this creates splitter for log content
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Create splitter with same settings as documentation splitter (500 chars with 50 overlap - could be tuned differently for logs but currently uses same configuration)
