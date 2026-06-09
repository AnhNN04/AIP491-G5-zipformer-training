import pytest
from unittest.mock import MagicMock, patch
from src.adapters.storage.minio_client import MinioStorageService

@pytest.fixture
def mock_minio():
    with patch("src.adapters.storage.minio_client.Minio") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance

@pytest.mark.anyio
async def test_minio_initialization(mock_minio):
    service = MinioStorageService(
        endpoint="localhost:9999",
        access_key="test_access",
        secret_key="test_secret",
        secure=True
    )
    assert service.endpoint == "localhost:9999"
    assert service.access_key == "test_access"
    assert service.secure is True

@pytest.mark.anyio
async def test_ensure_bucket(mock_minio):
    service = MinioStorageService()
    
    # Bucket exists
    mock_minio.bucket_exists.return_value = True
    await service.ensure_bucket("test-bucket")
    mock_minio.bucket_exists.assert_called_with("test-bucket")
    mock_minio.make_bucket.assert_not_called()

    # Bucket does not exist
    mock_minio.bucket_exists.reset_mock()
    mock_minio.bucket_exists.return_value = False
    await service.ensure_bucket("new-bucket")
    mock_minio.bucket_exists.assert_called_with("new-bucket")
    mock_minio.make_bucket.assert_called_with("new-bucket")

@pytest.mark.anyio
async def test_upload_file(mock_minio):
    service = MinioStorageService()
    mock_minio.bucket_exists.return_value = True
    
    result = await service.upload_file(
        bucket_name="test-bucket",
        object_name="audio.wav",
        data=b"fake audio data",
        content_type="audio/wav"
    )
    
    assert result == "s3://test-bucket/audio.wav"
    mock_minio.put_object.assert_called_once()
    args, kwargs = mock_minio.put_object.call_args
    assert kwargs["bucket_name"] == "test-bucket"
    assert kwargs["object_name"] == "audio.wav"
    assert kwargs["content_type"] == "audio/wav"
    assert kwargs["length"] == len(b"fake audio data")

@pytest.mark.anyio
async def test_download_file(mock_minio):
    service = MinioStorageService()
    mock_response = MagicMock()
    mock_response.read.return_value = b"downloaded audio data"
    mock_minio.get_object.return_value = mock_response

    result = await service.download_file("test-bucket", "audio.wav")
    assert result == b"downloaded audio data"
    mock_minio.get_object.assert_called_with("test-bucket", "audio.wav")
    mock_response.close.assert_called_once()
    mock_response.release_conn.assert_called_once()

@pytest.mark.anyio
async def test_get_presigned_url(mock_minio):
    service = MinioStorageService()
    mock_minio.presigned_get_object.return_value = "https://presigned-url.com/file"

    url = await service.get_presigned_url("test-bucket", "audio.wav", expires=1800)
    assert url == "https://presigned-url.com/file"
    mock_minio.presigned_get_object.assert_called_once()
    args, kwargs = mock_minio.presigned_get_object.call_args
    assert kwargs["bucket_name"] == "test-bucket"
    assert kwargs["object_name"] == "audio.wav"
    assert kwargs["expires"].total_seconds() == 1800

@pytest.mark.anyio
async def test_delete_file(mock_minio):
    service = MinioStorageService()
    await service.delete_file("test-bucket", "audio.wav")
    mock_minio.remove_object.assert_called_with("test-bucket", "audio.wav")
