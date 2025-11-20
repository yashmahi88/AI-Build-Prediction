from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def create_confluence_splitter() -> RecursiveCharacterTextSplitter:
    """Create splitter for documentation (larger chunks)"""
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def create_log_splitter() -> RecursiveCharacterTextSplitter:
    """Create splitter for logs (smaller chunks)"""
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
