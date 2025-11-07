import hashlib
from typing import Optional

def calculate_file_hash(filepath: str) -> str:
    """Generate MD5 hash for file content tracking"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_file_safely(filepath: str, encoding: str = 'utf-8') -> Optional[str]:
    """Read file with fallback encoding"""
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
