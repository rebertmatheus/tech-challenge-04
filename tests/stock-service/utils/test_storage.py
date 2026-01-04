import pytest
import pandas as pd
import numpy as np
import io
import joblib
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Adicionar o diretório stock-service ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stock-service'))

from utils.storage import (
    get_storage_client, load_hyperparameters, load_history_data,
    save_model, load_metrics, load_model, load_scaler, load_daily_data
)


class TestStorage:
    """Testes para o módulo storage.py"""

    @pytest.fixture
    def mock_container_client(self):
        """Mock container client do Azure Blob Storage"""
        return MagicMock()

    @pytest.fixture
    def sample_hyperparams(self):
        """Hiperparâmetros de exemplo"""
        return {
            "SEQUENCE_LENGTH": 30,
            "FEATURE_COLS": ["Close", "Volume", "High", "Low"],
            "TARGET_COL": "Close",
            "BATCH_SIZE": 32,
            "EPOCHS": 100
        }

    @pytest.fixture
    def sample_dataframe(self):
        """DataFrame de exemplo"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = {
            'Close': np.random.uniform(10, 50, 100),
            'Volume': np.random.uniform(1000, 10000, 100),
            'High': np.random.uniform(15, 55, 100),
            'Low': np.random.uniform(5, 45, 100)
        }
        return pd.DataFrame(data, index=dates)

    @patch('utils.storage.BlobServiceClient')
    @patch('utils.storage.logger')
    def test_get_storage_client_success(self, mock_logger, mock_blob_service_class):
        """Testa criação de container client com sucesso"""
        # Configurar mocks
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_blob_service_class.from_connection_string.return_value = mock_blob_service

        conn_str = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
        container_name = "test-container"

        result = get_storage_client(conn_str, container_name)

        assert result == mock_container_client
        mock_blob_service_class.from_connection_string.assert_called_once_with(conn_str)
        mock_blob_service.get_container_client.assert_called_once_with(container_name)
        mock_container_client.create_container.assert_called_once()
        mock_logger.info.assert_called_with(f"Container {container_name} criado")

    @patch('utils.storage.BlobServiceClient')
    @patch('utils.storage.logger')
    def test_get_storage_client_existing_container(self, mock_logger, mock_blob_service_class):
        """Testa criação de container client quando container já existe"""
        from azure.core.exceptions import ResourceExistsError

        # Configurar mocks
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_container_client.create_container.side_effect = ResourceExistsError("Container already exists")
        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_blob_service_class.from_connection_string.return_value = mock_blob_service

        conn_str = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
        container_name = "test-container"

        result = get_storage_client(conn_str, container_name)

        assert result == mock_container_client
        mock_logger.info.assert_called_with(f"Container {container_name} já existe")

    @patch('utils.storage.BlobServiceClient')
    def test_get_storage_client_connection_error(self, mock_blob_service_class):
        """Testa erro de conexão com Azure Blob Storage"""
        mock_blob_service_class.from_connection_string.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            get_storage_client("invalid_conn_str", "test-container")

    @patch('utils.storage.logger')
    def test_load_hyperparameters_success(self, mock_logger, mock_container_client, sample_hyperparams):
        """Testa carregamento de hiperparâmetros com sucesso"""
        # Configurar mock do blob
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        # Mock do download
        mock_download = MagicMock()
        mock_blob_data = MagicMock()
        mock_blob_data.decode.return_value = '{"SEQUENCE_LENGTH": 30, "FEATURE_COLS": ["Close", "Volume", "High", "Low"]}'
        mock_download.readall.return_value = mock_blob_data
        mock_blob_client.download_blob.return_value = mock_download

        ticker = "PETR4"
        result = load_hyperparameters(mock_container_client, ticker)

        expected = {"SEQUENCE_LENGTH": 30, "FEATURE_COLS": ["Close", "Volume", "High", "Low"]}
        assert result == expected
        mock_container_client.get_blob_client.assert_called_once_with("hyperparameters/PETR4.json")
        mock_logger.info.assert_called_once_with("Hiperparâmetros carregados: hyperparameters/PETR4.json")

    @patch('utils.storage.logger')
    def test_load_hyperparameters_file_not_found(self, mock_logger, mock_container_client):
        """Testa erro quando hiperparâmetros não existem"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Hiperparâmetros não encontrados para PETR4"):
            load_hyperparameters(mock_container_client, "PETR4")

    @patch('utils.storage.logger')
    def test_load_history_data_success(self, mock_logger, mock_container_client, sample_dataframe):
        """Testa carregamento de dados históricos com sucesso"""
        # Configurar mock do blob
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        # Mock do download e parquet
        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_parquet_data"
        mock_blob_client.download_blob.return_value = mock_download

        with patch('utils.storage.pd.read_parquet', return_value=sample_dataframe) as mock_read_parquet:
            result = load_history_data(mock_container_client, "PETR4")

            assert result.equals(sample_dataframe)
            mock_container_client.get_blob_client.assert_called_once_with("history/PETR4.parquet")
            mock_logger.info.assert_called_once()
            assert "history/PETR4.parquet" in mock_logger.info.call_args[0][0]
            assert "100 registros" in mock_logger.info.call_args[0][0]

    @patch('utils.storage.logger')
    def test_load_history_data_empty_dataframe(self, mock_logger, mock_container_client):
        """Testa erro quando dados históricos estão vazios"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_parquet_data"
        mock_blob_client.download_blob.return_value = mock_download

        empty_df = pd.DataFrame()
        with patch('utils.storage.pd.read_parquet', return_value=empty_df):
            with pytest.raises(ValueError, match="Dados históricos vazios para PETR4"):
                load_history_data(mock_container_client, "PETR4")

    @patch('utils.storage.logger')
    def test_save_model_success(self, mock_logger, mock_container_client):
        """Testa salvamento de modelo com sucesso"""
        # Configurar mocks dos blobs
        mock_model_blob = MagicMock()
        mock_scaler_blob = MagicMock()
        mock_metrics_blob = MagicMock()

        def get_blob_client_side_effect(path):
            if "PETR4_v1.ckpt" in path:
                return mock_model_blob
            elif "PETR4_v1_scaler.pkl" in path:
                return mock_scaler_blob
            elif "PETR4_v1_metrics.pkl" in path:
                return mock_metrics_blob

        mock_container_client.get_blob_client.side_effect = get_blob_client_side_effect

        model_bytes = b"fake_model_data"
        scaler_bytes = b"fake_scaler_data"
        metrics_bytes = b"fake_metrics_data"

        result = save_model(mock_container_client, "PETR4", "v1", model_bytes, scaler_bytes, metrics_bytes)

        expected_result = {
            "model_path": "models/PETR4_v1.ckpt",
            "scaler_path": "models/PETR4_v1_scaler.pkl",
            "metrics_path": "models/PETR4_v1_metrics.pkl"
        }
        assert result == expected_result

        # Verificar chamadas de upload
        mock_model_blob.upload_blob.assert_called_once_with(model_bytes, overwrite=True)
        mock_scaler_blob.upload_blob.assert_called_once_with(scaler_bytes, overwrite=True)
        mock_metrics_blob.upload_blob.assert_called_once_with(metrics_bytes, overwrite=True)

        # Verificar logs
        assert mock_logger.info.call_count == 3

    @patch('utils.storage.logger')
    def test_save_model_without_metrics(self, mock_logger, mock_container_client):
        """Testa salvamento de modelo sem métricas"""
        # Configurar mocks dos blobs
        mock_model_blob = MagicMock()
        mock_scaler_blob = MagicMock()

        def get_blob_client_side_effect(path):
            if "PETR4_v1.ckpt" in path:
                return mock_model_blob
            elif "PETR4_v1_scaler.pkl" in path:
                return mock_scaler_blob

        mock_container_client.get_blob_client.side_effect = get_blob_client_side_effect

        model_bytes = b"fake_model_data"
        scaler_bytes = b"fake_scaler_data"

        result = save_model(mock_container_client, "PETR4", "v1", model_bytes, scaler_bytes)

        expected_result = {
            "model_path": "models/PETR4_v1.ckpt",
            "scaler_path": "models/PETR4_v1_scaler.pkl"
        }
        assert result == expected_result
        assert "metrics_path" not in result

    @patch('utils.storage.logger')
    def test_load_metrics_success(self, mock_logger, mock_container_client):
        """Testa carregamento de métricas com sucesso"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_metrics_data"
        mock_blob_client.download_blob.return_value = mock_download

        expected_metrics = {
            'validacao': {'mae': 1.5, 'rmse': 2.1},
            'teste': {'mae': 1.8, 'rmse': 2.3}
        }

        with patch('utils.storage.joblib.load', return_value=expected_metrics) as mock_joblib_load:
            result = load_metrics(mock_container_client, "PETR4", "v1")

            assert result == expected_metrics
            mock_container_client.get_blob_client.assert_called_once_with("models/PETR4_v1_metrics.pkl")
            mock_logger.info.assert_called_once_with("Métricas carregadas: models/PETR4_v1_metrics.pkl")

    @patch('utils.storage.logger')
    def test_load_model_success(self, mock_logger, mock_container_client):
        """Testa carregamento de modelo com sucesso"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        expected_bytes = b"fake_model_bytes"
        mock_download.readall.return_value = expected_bytes
        mock_blob_client.download_blob.return_value = mock_download

        result = load_model(mock_container_client, "PETR4", "v1")

        assert result == expected_bytes
        mock_container_client.get_blob_client.assert_called_once_with("models/PETR4_v1.ckpt")
        mock_logger.info.assert_called_once()
        assert "Modelo carregado: models/PETR4_v1.ckpt" in mock_logger.info.call_args[0][0]

    @patch('utils.storage.logger')
    def test_load_scaler_success(self, mock_logger, mock_container_client):
        """Testa carregamento de scaler com sucesso"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_scaler_data"
        mock_blob_client.download_blob.return_value = mock_download

        expected_scalers = {
            'feature_scaler': MagicMock(),
            'target_scaler': MagicMock()
        }

        with patch('utils.storage.joblib.load', return_value=expected_scalers) as mock_joblib_load:
            result = load_scaler(mock_container_client, "PETR4", "v1")

            assert result == expected_scalers
            mock_container_client.get_blob_client.assert_called_once_with("models/PETR4_v1_scaler.pkl")
            mock_logger.info.assert_called_once_with("Scaler carregado: models/PETR4_v1_scaler.pkl")

    @patch('utils.storage.logger')
    def test_load_daily_data_success(self, mock_logger, mock_container_client, sample_dataframe):
        """Testa carregamento de dados diários com sucesso"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_parquet_data"
        mock_blob_client.download_blob.return_value = mock_download

        # Data de exemplo
        from datetime import datetime
        test_date = datetime(2023, 1, 15)

        with patch('utils.storage.pd.read_parquet', return_value=sample_dataframe.head(1)) as mock_read_parquet:
            result = load_daily_data(mock_container_client, "PETR4", test_date)

            assert len(result) == 1
            mock_container_client.get_blob_client.assert_called_once_with("2023/01/15/PETR4.parquet")
            mock_logger.info.assert_called_once()
            assert "2023/01/15/PETR4.parquet" in mock_logger.info.call_args[0][0]

    @patch('utils.storage.logger')
    def test_load_daily_data_string_date(self, mock_logger, mock_container_client, sample_dataframe):
        """Testa carregamento de dados diários com data como string"""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = True

        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake_parquet_data"
        mock_blob_client.download_blob.return_value = mock_download

        with patch('utils.storage.pd.read_parquet', return_value=sample_dataframe.head(1)) as mock_read_parquet:
            result = load_daily_data(mock_container_client, "PETR4", "2023-01-15")

            assert len(result) == 1
            mock_container_client.get_blob_client.assert_called_once_with("2023/01/15/PETR4.parquet")