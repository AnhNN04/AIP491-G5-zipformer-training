import io
import asyncio
import os
from typing import Optional
from minio import Minio
from src.domain.interfaces import IStorageService

class MinioStorageService(IStorageService):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: Optional[bool] = None
    ):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minio_admin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minio_password")
        
        # Parse secure flag
        if secure is None:
            secure_env = os.getenv("MINIO_SECURE", "False")
            self.secure = secure_env.lower() in ("true", "1", "yes")
        else:
            self.secure = secure

        # Initialize the synchronous MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def _ensure_bucket(self, bucket_name: str) -> None:
        """Synchronous helper to verify and create bucket."""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    async def ensure_bucket(self, bucket_name: str) -> None:
        """Asynchronously ensure a bucket exists."""
        await asyncio.to_thread(self._ensure_bucket, bucket_name)

    async def upload_file(self, bucket_name: str, object_name: str, data: bytes, content_type: str) -> str:
        """
        Uploads binary data asynchronously. Returns the storage path/URI.
        """
        # Ensure the bucket exists first
        await self.ensure_bucket(bucket_name)
        
        data_stream = io.BytesIO(data)
        length = len(data)

        def _upload():
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=length,
                content_type=content_type
            )
            return f"s3://{bucket_name}/{object_name}"

        return await asyncio.to_thread(_upload)

    async def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """
        Downloads binary data asynchronously.
        """
        def _download():
            response = self.client.get_object(bucket_name, object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(_download)

    async def get_presigned_url(self, bucket_name: str, object_name: str, expires: int = 3600) -> str:
        """
        Generates a presigned download URL asynchronously.
        """
        def _get_url():
            # timedelta-based expiry
            from datetime import timedelta
            return self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )

        return await asyncio.to_thread(_get_url)

    async def delete_file(self, bucket_name: str, object_name: str) -> None:
        """
        Deletes an object asynchronously.
        """
        def _delete():
            self.client.remove_object(bucket_name, object_name)

        await asyncio.to_thread(_delete)
