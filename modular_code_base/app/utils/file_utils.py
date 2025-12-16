import hashlib  # Cryptographic hashing library for generating MD5 checksums of file content
from typing import Optional  # Type hint for optional return values that can be None


def calculate_file_hash(filepath: str) -> str:  # Function to generate MD5 hash of file content for change detection and deduplication
    """Generate MD5 hash for file content tracking"""  # Docstring explaining this creates MD5 checksum to track if file has changed
    hash_md5 = hashlib.md5()  # Create MD5 hash object (MD5 is fast and sufficient for file change detection, not cryptographic security)
    with open(filepath, "rb") as f:  # Open file in binary read mode (rb) using context manager for automatic cleanup
        for chunk in iter(lambda: f.read(4096), b""):  # Read file in 4KB chunks using iterator pattern (lambda: f.read(4096) reads chunk, b"" is sentinel value to stop iteration)
            hash_md5.update(chunk)  # Update hash with each chunk (incremental hashing avoids loading entire file into memory)
    return hash_md5.hexdigest()  # Return hex string representation of hash (e.g., "5d41402abc4b2a76b9719d911017c592")


def read_file_safely(filepath: str, encoding: str = 'utf-8') -> Optional[str]:  # Function to read file with automatic encoding fallback to handle various text file formats
    """Read file with fallback encoding"""  # Docstring explaining this tries UTF-8 first, then Latin-1 if UTF-8 fails
    try:  # First attempt: try UTF-8 encoding (most common for modern files)
        with open(filepath, 'r', encoding=encoding) as f:  # Open file in text read mode with specified encoding (default UTF-8) using context manager
            return f.read()  # Read entire file content as string and return
    except UnicodeDecodeError:  # Catch encoding error if file contains bytes that aren't valid UTF-8
        with open(filepath, 'r', encoding='latin-1') as f:  # Retry with Latin-1 encoding (also known as ISO-8859-1, accepts any byte value so never fails)
            return f.read()  # Read entire file content with fallback encoding and return
    except Exception as e:  # Catch any other errors (file not found, permission denied, etc.)
        print(f"Error reading {filepath}: {e}")  # Print error message to console with filepath and exception details
        return None  # Return None to indicate read failure (caller can check for None and handle gracefully)
