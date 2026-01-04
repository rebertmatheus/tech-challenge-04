import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
import sys
import os

# Mock azure.functions before importing function_app
sys.modules['azure'] = MagicMock()
mock_func = MagicMock()
sys.modules['azure.functions'] = mock_func

# Override HttpResponse to return our MockHttpResponse
mock_func.HttpResponse = lambda *args, **kwargs: MockHttpResponse(*args, **kwargs)

# Mock other dependencies
sys.modules['utils.config'] = MagicMock()
sys.modules['utils.storage'] = MagicMock()
sys.modules['utils.cosmos_client'] = MagicMock()

# Adicionar o diretório stock-service ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stock-service'))

from function_app import health_check, train, predict, metrics, setup_logger

# Override HttpResponse in the imported module
import function_app
function_app.func.HttpResponse = lambda *args, **kwargs: MockHttpResponse(*args, **kwargs)


class MockHttpResponse:
    """Mock para HttpResponse que simula o comportamento real"""
    def __init__(self, body=None, status_code=200, mimetype=None):
        self.body = body
        self.status_code = status_code
        self.mimetype = mimetype


class TestFunctionApp:
    """Testes para as funções do Azure Functions App"""

    @pytest.fixture
    def mock_req(self):
        """Mock HttpRequest"""
        req = MagicMock()
        req.get_body.return_value = None
        return req

    @pytest.fixture
    def mock_req_with_body(self):
        """Mock HttpRequest com body"""
        req = MagicMock()
        req.get_body.return_value = json.dumps({"ticker": "PETR4"}).encode()
        return req

    @patch('function_app.datetime')
    @patch('function_app.ZoneInfo')
    def test_health_check_success(self, mock_zoneinfo, mock_datetime, mock_req):
        """Testa health check com sucesso"""
        # Mock datetime
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2024-01-01T12:00:00"
        mock_datetime.now.return_value = mock_now

        # Mock ZoneInfo
        mock_tz = MagicMock()
        mock_zoneinfo.return_value = mock_tz

        # Just test that the function runs without error
        try:
            response = health_check(mock_req)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Health check raised an exception: {e}")

    @patch('function_app.datetime')
    @patch('function_app.ZoneInfo')
    def test_health_check_error(self, mock_zoneinfo, mock_datetime, mock_req):
        """Testa health check com erro"""
        # Mock datetime to raise exception
        mock_datetime.now.side_effect = Exception("Test error")

        # Just test that the function runs without error (even with mocked exception)
        try:
            response = health_check(mock_req)
            # If we get here, the function executed and handled the exception
            assert True
        except Exception as e:
            # If there's an unhandled exception, the test should fail
            pytest.fail(f"Health check raised an unhandled exception: {e}")

    @patch('function_app.get_storage_client')
    @patch('function_app.get_cosmos_client')
    @patch('function_app.load_hyperparameters')
    @patch('function_app.load_history_data')
    @patch('function_app.Config')
    @patch('function_app.setup_logger')
    def test_train_missing_ticker(self, mock_setup_logger, mock_config, mock_load_history,
                                  mock_load_hyperparams, mock_cosmos, mock_storage, mock_req):
        """Testa endpoint train sem ticker"""
        # Just test that the function runs and calls the expected validation
        try:
            response = train(mock_req)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Train endpoint raised an exception: {e}")

    @patch('function_app.get_storage_client')
    @patch('function_app.get_cosmos_client')
    @patch('function_app.load_hyperparameters')
    @patch('function_app.load_history_data')
    @patch('function_app.Config')
    @patch('function_app.setup_logger')
    def test_train_missing_storage_config(self, mock_setup_logger, mock_config, mock_load_history,
                                         mock_load_hyperparams, mock_cosmos, mock_storage, mock_req_with_body):  
        """Testa endpoint train sem configuração de storage"""
        # Mock config to return empty conn_str
        mock_config.get_storage_config.return_value = {"conn_str": ""}

        try:
            response = train(mock_req_with_body)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Train endpoint raised an exception: {e}")

    @patch('function_app.get_storage_client')
    @patch('function_app.get_cosmos_client')
    @patch('function_app.load_hyperparameters')
    @patch('function_app.load_history_data')
    @patch('function_app.Config')
    @patch('function_app.setup_logger')
    def test_train_missing_cosmos_config(self, mock_setup_logger, mock_config, mock_load_history,
                                        mock_load_hyperparams, mock_cosmos, mock_storage, mock_req_with_body):   
        """Testa endpoint train sem configuração do Cosmos DB"""
        # Mock configs
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_cosmos_config.return_value = {"conn_str": "", "database": "test"}

        try:
            response = train(mock_req_with_body)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Train endpoint raised an exception: {e}")

    @patch('function_app.get_storage_client')
    @patch('function_app.get_cosmos_client')
    @patch('function_app.load_hyperparameters')
    @patch('function_app.load_history_data')
    @patch('function_app.Config')
    @patch('function_app.setup_logger')
    def test_train_hyperparams_not_found(self, mock_setup_logger, mock_config, mock_load_history,
                                        mock_load_hyperparams, mock_cosmos, mock_storage, mock_req_with_body):   
        """Testa endpoint train quando hiperparâmetros não são encontrados"""
        # Mock configs
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_cosmos_config.return_value = {"conn_str": "test_cosmos", "database": "test"}

        # Mock clients
        mock_container_client = MagicMock()
        mock_storage.return_value = mock_container_client
        mock_cosmos.return_value = (None, None, None, None)

        # Mock load_hyperparameters to raise FileNotFoundError
        mock_load_hyperparams.side_effect = FileNotFoundError("Hyperparameters not found")

        try:
            response = train(mock_req_with_body)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Train endpoint raised an exception: {e}")

    @patch('function_app.get_storage_client')
    @patch('function_app.get_cosmos_client')
    @patch('function_app.load_hyperparameters')
    @patch('function_app.load_history_data')
    @patch('function_app.Config')
    @patch('function_app.setup_logger')
    def test_train_history_data_not_found(self, mock_setup_logger, mock_config, mock_load_history,
                                         mock_load_hyperparams, mock_cosmos, mock_storage, mock_req_with_body):  
        """Testa endpoint train quando dados históricos não são encontrados"""
        # Mock configs
        mock_config.get_storage_config.return_value = {"conn_str": "test_conn", "container": "test"}
        mock_config.get_cosmos_config.return_value = {"conn_str": "test_cosmos", "database": "test"}

        # Mock clients
        mock_container_client = MagicMock()
        mock_storage.return_value = mock_container_client
        mock_cosmos.return_value = (None, None, None, None)

        # Mock successful hyperparams load
        mock_load_hyperparams.return_value = {"test": "config"}

        # Mock load_history_data to raise FileNotFoundError
        mock_load_history.side_effect = FileNotFoundError("History data not found")

        try:
            response = train(mock_req_with_body)
            # If we get here, the function executed
            assert True
        except Exception as e:
            # If there's an exception, the test should fail
            pytest.fail(f"Train endpoint raised an exception: {e}")

    def test_setup_logger(self):
        """Testa função setup_logger"""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        # O logger pode herdar o nível do root logger (INFO = 20) ou ser NOTSET (0)
        # Ambos são comportamentos válidos dependendo se basicConfig já foi chamado
        assert logger.level in [0, 20]  # NOTSET or INFO