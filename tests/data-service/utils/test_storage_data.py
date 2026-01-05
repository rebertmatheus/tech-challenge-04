import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Load the storage module dynamically
storage_path = Path(__file__).resolve().parents[3] / "data-service" / "utils" / "storage.py"
spec = importlib.util.spec_from_file_location("storage", str(storage_path))
storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage_module)
get_storage_client = storage_module.get_storage_client


class TestGetStorageClient:

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    @patch('logging.getLogger')
    def test_get_storage_client_success_new_container(self, mock_get_logger, mock_from_connection_string):
        """Test successful creation of new container"""
        # Mock the logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Mock the blob service
        mock_blob_service = Mock()
        mock_from_connection_string.return_value = mock_blob_service

        # Mock the container client
        mock_container_client = Mock()
        mock_blob_service.get_container_client.return_value = mock_container_client

        conn_str = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;BlobEndpoint=https://test.blob.core.windows.net/"
        container_name = "test-container"

        result = get_storage_client(conn_str, container_name)

        # Assertions
        mock_from_connection_string.assert_called_once_with(conn_str)
        mock_blob_service.get_container_client.assert_called_once_with(container_name)
        mock_container_client.create_container.assert_called_once()
        mock_logger.info.assert_called_once_with(f"Container {container_name} criado")
        assert result == mock_container_client

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    @patch('logging.getLogger')
    def test_get_storage_client_container_already_exists(self, mock_get_logger, mock_from_connection_string):
        """Test when container already exists"""
        from azure.core.exceptions import ResourceExistsError

        # Mock the logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Mock the blob service
        mock_blob_service = Mock()
        mock_from_connection_string.return_value = mock_blob_service

        # Mock the container client
        mock_container_client = Mock()
        mock_blob_service.get_container_client.return_value = mock_container_client

        # Mock container creation raising ResourceExistsError
        mock_container_client.create_container.side_effect = ResourceExistsError("Container already exists")

        conn_str = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;BlobEndpoint=https://test.blob.core.windows.net/"
        container_name = "existing-container"

        result = get_storage_client(conn_str, container_name)

        # Assertions
        mock_from_connection_string.assert_called_once_with(conn_str)
        mock_blob_service.get_container_client.assert_called_once_with(container_name)
        mock_container_client.create_container.assert_called_once()
        mock_logger.info.assert_called_once_with(f"Container {container_name} já existe")
        assert result == mock_container_client

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    @patch('logging.getLogger')
    def test_get_storage_client_connection_failure(self, mock_get_logger, mock_from_connection_string):
        """Test failure in blob service connection"""
        # Mock the logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Mock from_connection_string to raise exception
        mock_from_connection_string.side_effect = Exception("Connection failed")

        conn_str = "invalid-connection-string"
        container_name = "test-container"

        with pytest.raises(Exception, match="Connection failed"):
            get_storage_client(conn_str, container_name)

        # Assertions
        mock_from_connection_string.assert_called_once_with(conn_str)
        mock_logger.exception.assert_called_once_with("Falha ao conectar no Azure Blob Storage")

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    @patch('logging.getLogger')
    def test_get_storage_client_create_container_failure(self, mock_get_logger, mock_from_connection_string):
        """Test failure in container creation (not ResourceExistsError)"""
        # Mock the logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Mock the blob service
        mock_blob_service = Mock()
        mock_from_connection_string.return_value = mock_blob_service

        # Mock the container client
        mock_container_client = Mock()
        mock_blob_service.get_container_client.return_value = mock_container_client

        # Mock container creation raising unexpected exception
        mock_container_client.create_container.side_effect = Exception("Unexpected error")

        conn_str = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;BlobEndpoint=https://test.blob.core.windows.net/"
        container_name = "test-container"

        with pytest.raises(Exception, match="Unexpected error"):
            get_storage_client(conn_str, container_name)

        # Assertions
        mock_from_connection_string.assert_called_once_with(conn_str)
        mock_blob_service.get_container_client.assert_called_once_with(container_name)
        mock_container_client.create_container.assert_called_once()
        mock_logger.exception.assert_called_once_with("Falha ao conectar no Azure Blob Storage")