import sys
from pathlib import Path
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import importlib.util
import pandas as pd

# Load modules dynamically to avoid import issues
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules
base_path = Path(__file__).resolve().parents[2] / "data-service"

# Load utils modules first
config_path = base_path / "utils" / "config.py"
config_module = load_module("utils.config", config_path)

storage_path = base_path / "utils" / "storage.py"
storage_module = load_module("utils.storage", storage_path)

yfinance_path = base_path / "utils" / "yfinance_client.py"
yfinance_module = load_module("utils.yfinance_client", yfinance_path)

parquet_path = base_path / "utils" / "parquet_handler.py"
parquet_module = load_module("utils.parquet_handler", parquet_path)

feature_path = base_path / "utils" / "feature_engineering.py"
feature_module = load_module("utils.feature_engineering", feature_path)

# Now load function_app
function_app_path = base_path / "function_app.py"
function_app_module = load_module("function_app", function_app_path)

# Import the functions we want to test
health_check = function_app_module.health_check
fetch_day = function_app_module.fetch_day
fetch_history = function_app_module.fetch_history


class TestHealthCheck:
    """Testes para a função health_check"""

    def test_health_check_success(self):
        """Testa health check bem-sucedido"""
        # Mock HttpRequest
        mock_req = Mock()

        # Call the function
        response = health_check(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "data-service"
        assert "timestamp" in response_data


class TestFetchDay:
    """Testes para a função fetch_day"""

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    @patch('function_app.datetime')
    def test_fetch_day_success_no_date_param(self, mock_datetime, mock_engineer, mock_parquet,
                                           mock_yf_client, mock_storage_client, mock_config):
        """Testa fetch_day bem-sucedido sem parâmetro de data"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_loopback_period.return_value = 30

        # Mock datetime
        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        mock_datetime.now.return_value = now

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        mock_parquet_handler = Mock()
        mock_parquet.return_value = mock_parquet_handler

        mock_engineer_instance = Mock()
        mock_engineer.return_value = mock_engineer_instance

        # Mock data fetching
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [10.0, 11.0],
            'Volume': [1000, 1100],
            'High': [10.5, 11.5],
            'Low': [9.5, 10.5]
        })
        mock_yf.fetch_ticker_data.return_value = mock_df
        mock_engineer_instance.create_features.return_value = mock_df
        mock_parquet_handler.save_daily_data.return_value = "test_path"

        # Mock HttpRequest
        mock_req = Mock()
        mock_req.params = {}

        # Call the function
        response = fetch_day(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    @patch('function_app.datetime')
    def test_fetch_day_yfinance_returns_empty(self, mock_datetime, mock_engineer, mock_parquet,
                                             mock_yf_client, mock_storage_client, mock_config):
        """Testa fetch_day quando yfinance retorna dataframe vazio"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_loopback_period.return_value = 30

        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        mock_datetime.now.return_value = now

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        # Mock empty dataframe
        mock_yf.fetch_ticker_data.return_value = pd.DataFrame()

        # Mock HttpRequest
        mock_req = Mock()
        mock_req.params = {}

        # Call the function
        response = fetch_day(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["status"] == "completed"
        assert response_data["successful"] == 0
        assert response_data["failed"] == 1
        assert "Sem dados" in response_data["results"][0]

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    @patch('function_app.datetime')
    def test_fetch_day_create_features_returns_none(self, mock_datetime, mock_engineer, mock_parquet,
                                                   mock_yf_client, mock_storage_client, mock_config):
        """Testa fetch_day quando create_features retorna None"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_loopback_period.return_value = 30

        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        mock_datetime.now.return_value = now

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        mock_parquet_handler = Mock()
        mock_parquet.return_value = mock_parquet_handler

        mock_engineer_instance = Mock()
        mock_engineer.return_value = mock_engineer_instance

        # Mock data fetching
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [10.0, 11.0],
            'Volume': [1000, 1100],
            'High': [10.5, 11.5],
            'Low': [9.5, 10.5]
        })
        mock_yf.fetch_ticker_data.return_value = mock_df
        mock_engineer_instance.create_features.return_value = None  # Return None

        # Mock HttpRequest
        mock_req = Mock()
        mock_req.params = {}

        # Call the function
        response = fetch_day(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["status"] == "completed"
        assert response_data["successful"] == 0
        assert response_data["failed"] == 0  # None is logged but not counted as failed

    @patch('function_app.Config')
    def test_fetch_day_missing_storage_config(self, mock_config):
        """Testa fetch_day com configuração de storage faltando"""
        mock_config.get_storage_config.return_value = {"conn_str": "", "container": "test"}

        mock_req = Mock()
        response = fetch_day(mock_req)

        assert response.status_code == 500
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "AzureWebJobsStorage não definido" in response_data["error"]

    @patch('function_app.Config')
    def test_fetch_day_value_error_config(self, mock_config):
        """Testa fetch_day com ValueError na configuração"""
        mock_config.get_tickers.side_effect = ValueError("Config error")

        mock_req = Mock()
        response = fetch_day(mock_req)

        assert response.status_code == 400
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "Config error" in response_data["error"]

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    @patch('function_app.datetime')
    def test_fetch_day_unexpected_error(self, mock_datetime, mock_engineer, mock_parquet,
                                       mock_yf_client, mock_storage_client, mock_config):
        """Testa fetch_day com erro inesperado"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_loopback_period.return_value = 30

        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        mock_datetime.now.return_value = now

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        # Mock unexpected error
        mock_yf.fetch_ticker_data.side_effect = Exception("Unexpected error")

        # Mock HttpRequest
        mock_req = Mock()
        mock_req.params = {}

        # Call the function
        response = fetch_day(mock_req)

        # Verify error response
        assert response.status_code == 200  # Function returns 200 even with processing errors
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["status"] == "completed"
        assert response_data["successful"] == 0
        assert response_data["failed"] == 1
        assert "Unexpected error" in response_data["results"][0]


class TestFetchHistory:
    """Testes para a função fetch_history"""

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    def test_fetch_history_success(self, mock_engineer, mock_parquet, mock_yf_client,
                                 mock_storage_client, mock_config):
        """Testa fetch_history bem-sucedido"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_date_range.return_value = {"start": "2020-01-01", "end": "2024-01-01"}

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        mock_parquet_handler = Mock()
        mock_parquet.return_value = mock_parquet_handler

        mock_engineer_instance = Mock()
        mock_engineer.return_value = mock_engineer_instance

        # Mock data fetching
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [10.0, 11.0, 12.0],
            'Volume': [1000, 1100, 1200],
            'High': [10.5, 11.5, 12.5],
            'Low': [9.5, 10.5, 11.5]
        })
        mock_yf.fetch_ticker_data.return_value = mock_df
        mock_engineer_instance.create_features.return_value = mock_df

        # Mock HttpRequest
        mock_req = Mock()

        # Call the function
        response = fetch_history(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["success"] is True

    @patch('function_app.Config')
    def test_fetch_history_missing_storage_config(self, mock_config):
        """Testa fetch_history com configuração de storage faltando"""
        mock_config.get_storage_config.return_value = {"conn_str": "", "container": "test"}

        mock_req = Mock()
        response = fetch_history(mock_req)

        assert response.status_code == 500
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "AzureWebJobsStorage não definido" in response_data["error"]

    @patch('function_app.Config')
    def test_fetch_history_value_error_config(self, mock_config):
        """Testa fetch_history com ValueError na configuração"""
        mock_config.get_tickers.side_effect = ValueError("Config error")

        mock_req = Mock()
        response = fetch_history(mock_req)

        assert response.status_code == 400
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "Config error" in response_data["error"]

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    def test_fetch_history_unexpected_error(self, mock_engineer, mock_parquet, mock_yf_client,
                                          mock_storage_client, mock_config):
        """Testa fetch_history com erro inesperado"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_date_range.return_value = {"start": "2020-01-01", "end": "2024-01-01"}

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        # Mock unexpected error in config
        mock_config.get_date_range.side_effect = Exception("Unexpected config error")

        mock_req = Mock()
        response = fetch_history(mock_req)

        assert response.status_code == 500
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "Unexpected config error" in response_data["error"]
    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.FeatureEngineer')
    def test_fetch_history_processing_error(self, mock_engineer, mock_yf_client,
                                          mock_storage_client, mock_config):
        """Testa fetch_history com erro no processamento"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_date_range.return_value = {"start": "2020-01-01", "end": "2024-01-01"}

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        # Mock data fetching to raise exception
        mock_yf.fetch_ticker_data.side_effect = Exception("API Error")

        mock_req = Mock()
        response = fetch_history(mock_req)

        assert response.status_code == 500
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert "API Error" in response_data["error"]

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    def test_fetch_history_yfinance_returns_empty(self, mock_engineer, mock_parquet, mock_yf_client,
                                                 mock_storage_client, mock_config):
        """Testa fetch_history quando yfinance retorna dataframe vazio"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_date_range.return_value = {"start": "2020-01-01", "end": "2024-01-01"}

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        # Mock empty dataframe
        mock_yf.fetch_ticker_data.return_value = pd.DataFrame()

        # Mock HttpRequest
        mock_req = Mock()

        # Call the function
        response = fetch_history(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["success"] is True

    @patch('function_app.Config')
    @patch('function_app.get_storage_client')
    @patch('function_app.YFinanceClient')
    @patch('function_app.ParquetHandler')
    @patch('function_app.FeatureEngineer')
    def test_fetch_history_create_features_returns_none(self, mock_engineer, mock_parquet, mock_yf_client,
                                                       mock_storage_client, mock_config):
        """Testa fetch_history quando create_features retorna None"""
        # Setup mocks
        mock_config.get_tickers.return_value = ["PETR4"]
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_date_range.return_value = {"start": "2020-01-01", "end": "2024-01-01"}

        # Mock clients
        mock_container_client = Mock()
        mock_storage_client.return_value = mock_container_client

        mock_yf = Mock()
        mock_yf_client.return_value = mock_yf

        mock_parquet_handler = Mock()
        mock_parquet.return_value = mock_parquet_handler

        mock_engineer_instance = Mock()
        mock_engineer.return_value = mock_engineer_instance

        # Mock data fetching
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [10.0, 11.0, 12.0],
            'Volume': [1000, 1100, 1200],
            'High': [10.5, 11.5, 12.5],
            'Low': [9.5, 10.5, 11.5]
        })
        mock_yf.fetch_ticker_data.return_value = mock_df
        mock_engineer_instance.create_features.return_value = None  # Return None

        # Mock HttpRequest
        mock_req = Mock()

        # Call the function
        response = fetch_history(mock_req)

        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.get_body().decode('utf-8'))
        assert response_data["success"] is True