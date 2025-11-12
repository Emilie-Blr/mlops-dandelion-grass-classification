"""
MinIO S3 client utilities
"""

from pathlib import Path
from typing import Optional, Union
import io

from minio import Minio
from minio.error import S3Error
from loguru import logger
from PIL import Image

from src.config import settings


class MinIOClient:
    """MinIO client for S3 operations"""
    
    def __init__(self):
        """Initialize MinIO client"""
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self._ensure_buckets()
    
    def _ensure_buckets(self):
        """Ensure required buckets exist"""
        buckets = [
            settings.S3_BUCKET_MODELS,
            settings.S3_BUCKET_DATA,
            settings.S3_BUCKET_ARTIFACTS
        ]
        
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Error ensuring bucket {bucket}: {e}")
    
    def upload_file(
        self, 
        bucket_name: str, 
        object_name: str, 
        file_path: Union[str, Path]
    ) -> bool:
        """
        Upload a file to MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            file_path: Path to the file to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.fput_object(bucket_name, object_name, str(file_path))
            logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def upload_image(
        self, 
        bucket_name: str, 
        object_name: str, 
        image: Image.Image
    ) -> bool:
        """
        Upload a PIL Image to MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            image: PIL Image to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Upload
            self.client.put_object(
                bucket_name,
                object_name,
                img_bytes,
                length=img_bytes.getbuffer().nbytes,
                content_type='image/jpeg'
            )
            logger.info(f"Uploaded image to {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"Error uploading image: {e}")
            return False
    
    def download_file(
        self, 
        bucket_name: str, 
        object_name: str, 
        file_path: Union[str, Path]
    ) -> bool:
        """
        Download a file from MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            file_path: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.fget_object(bucket_name, object_name, str(file_path))
            logger.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            return True
        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def get_image(self, bucket_name: str, object_name: str) -> Optional[Image.Image]:
        """
        Get an image from MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            
        Returns:
            PIL Image or None if download fails
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            image = Image.open(io.BytesIO(response.data))
            return image
        except S3Error as e:
            logger.error(f"Error getting image: {e}")
            return None
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> list:
        """
        List objects in a bucket
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix filter
            
        Returns:
            List of object names
        """
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """
        Delete an object from MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"Deleted {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting object: {e}")
            return False


# Global client instance (lazy initialization)
_minio_client = None

def get_minio_client():
    """Get or create the global MinIO client instance"""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient()
    return _minio_client

# For backward compatibility
minio_client = get_minio_client()
