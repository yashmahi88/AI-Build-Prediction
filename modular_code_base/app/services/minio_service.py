"""MinIO/S3 operations service"""
import boto3
from typing import List, Dict, Optional
from app.core.config import get_settings

settings = get_settings()

class MinIOService:
    """Manages MinIO/S3 operations"""
    
    def __init__(self):
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=settings.minio_endpoint,
                aws_access_key_id=settings.minio_access_key,
                aws_secret_access_key=settings.minio_secret_key,
                verify=False
            )
            self.bucket = settings.minio_bucket
        except Exception as e:
            print(f"Warning: MinIO not initialized: {e}")
            self.client = None
    
    def list_files(self, prefix: str = '') -> List[Dict]:
        """List files in MinIO bucket"""
        if self.client is None:
            return []
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def get_file_content(self, key: str) -> Optional[str]:
        """Get file content from MinIO"""
        if self.client is None:
            return None
        
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error getting file: {e}")
            return None
