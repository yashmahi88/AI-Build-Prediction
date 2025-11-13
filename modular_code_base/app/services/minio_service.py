"""MinIO/S3 operations service"""
import boto3
import logging
from typing import List, Dict, Optional
from app.core.config import get_settings

logger = logging.getLogger(__name__)
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
            logger.warning(f"MinIO not initialized: {e}")
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
            logger.error(f"Error listing files: {e}")
            return []
    
    def list_objects(self, prefix: str = ""):
        """List all objects in the bucket (for file watcher)"""
        if self.client is None:
            return []
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            # Convert to object-like structure with attributes
            class MinIOObject:
                def __init__(self, key, etag, last_modified, size):
                    self.object_name = key
                    self.etag = etag
                    self.last_modified = last_modified
                    self.size = size
            
            objects = []
            for item in response.get('Contents', []):
                obj = MinIOObject(
                    key=item['Key'],
                    etag=item.get('ETag', ''),
                    last_modified=item.get('LastModified'),
                    size=item.get('Size', 0)
                )
                objects.append(obj)
            
            return objects
            
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def get_file_content(self, key: str) -> Optional[str]:
        """Get file content from MinIO"""
        if self.client is None:
            return None
        
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error getting file: {e}")
            return None

# """MinIO/S3 operations service"""
# import boto3
# from typing import List, Dict, Optional
# from app.core.config import get_settings

# settings = get_settings()

# class MinIOService:
#     """Manages MinIO/S3 operations"""
    
#     def __init__(self):
#         try:
#             self.client = boto3.client(
#                 's3',
#                 endpoint_url=settings.minio_endpoint,
#                 aws_access_key_id=settings.minio_access_key,
#                 aws_secret_access_key=settings.minio_secret_key,
#                 verify=False
#             )
#             self.bucket = settings.minio_bucket
#         except Exception as e:
#             print(f"Warning: MinIO not initialized: {e}")
#             self.client = None
    
#     def list_files(self, prefix: str = '') -> List[Dict]:
#         """List files in MinIO bucket"""
#         if self.client is None:
#             return []
        
#         try:
#             response = self.client.list_objects_v2(
#                 Bucket=self.bucket,
#                 Prefix=prefix
#             )
#             return response.get('Contents', [])
#         except Exception as e:
#             print(f"Error listing files: {e}")
#             return []
    
#     def get_file_content(self, key: str) -> Optional[str]:
#         """Get file content from MinIO"""
#         if self.client is None:
#             return None
        
#         try:
#             response = self.client.get_object(Bucket=self.bucket, Key=key)
#             return response['Body'].read().decode('utf-8', errors='ignore')
#         except Exception as e:
#             print(f"Error getting file: {e}")
#             return None
