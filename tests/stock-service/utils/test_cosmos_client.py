import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import importlib.util
import types

# Mock azure.cosmos before importing the module
class CosmosResourceNotFoundError(Exception):
    pass

mock_exceptions = types.ModuleType('exceptions')
mock_exceptions.CosmosResourceNotFoundError = CosmosResourceNotFoundError

with patch.dict('sys.modules', {
    'azure': MagicMock(),
    'azure.cosmos': MagicMock(),
    'azure.cosmos.exceptions': mock_exceptions,
    'zoneinfo': MagicMock(),
}):
    # Load the cosmos_client module dynamically
    cosmos_client_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "cosmos_client.py"
    spec = importlib.util.spec_from_file_location("cosmos_client", str(cosmos_client_path))
    cosmos_client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cosmos_client_module)
    
    # Override the exceptions in the module
    cosmos_client_module.exceptions = mock_exceptions


@patch.object(cosmos_client_module, 'CosmosClient')
def test_get_cosmos_client_success(mock_cosmos_client_class):
    """Test successful creation of Cosmos client and containers"""
    # Mock the client
    mock_client = MagicMock()
    mock_cosmos_client_class.from_connection_string.return_value = mock_client
    
    # Mock database
    mock_database = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_database
    
    # Mock containers
    mock_container_mv = MagicMock()
    mock_container_pred = MagicMock()
    mock_database.create_container_if_not_exists.side_effect = [mock_container_mv, mock_container_pred]
    
    # Call the function
    client, db, mv, pred = cosmos_client_module.get_cosmos_client("conn_str", "test_db")
    
    # Assertions
    mock_cosmos_client_class.from_connection_string.assert_called_once_with("conn_str")
    mock_client.create_database_if_not_exists.assert_called_once_with(id="test_db")
    assert mock_database.create_container_if_not_exists.call_count == 2
    assert client == mock_client
    assert db == mock_database
    assert mv == mock_container_mv
    assert pred == mock_container_pred


@patch.object(cosmos_client_module, 'CosmosClient')
def test_get_cosmos_client_database_error(mock_cosmos_client_class):
    """Test error when creating database"""
    mock_client = MagicMock()
    mock_cosmos_client_class.from_connection_string.return_value = mock_client
    mock_client.create_database_if_not_exists.side_effect = Exception("DB error")
    
    with pytest.raises(Exception, match="DB error"):
        cosmos_client_module.get_cosmos_client("conn_str", "test_db")


@patch.object(cosmos_client_module, 'CosmosClient')
def test_get_cosmos_client_container_error(mock_cosmos_client_class):
    """Test error when creating predictions container (second container)"""
    mock_client = MagicMock()
    mock_cosmos_client_class.from_connection_string.return_value = mock_client
    
    mock_database = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_database
    
    # Mock containers - first (model_versions) succeeds, second (predictions) fails
    mock_container_mv = MagicMock()
    mock_database.create_container_if_not_exists.side_effect = [mock_container_mv, Exception("Container error")]
    
    with pytest.raises(Exception, match="Container error"):
        cosmos_client_module.get_cosmos_client("conn_str", "test_db")


@patch.object(cosmos_client_module, 'CosmosClient')
def test_get_cosmos_client_model_versions_container_error(mock_cosmos_client_class):
    """Test error when creating model_versions container (first container)"""
    mock_client = MagicMock()
    mock_cosmos_client_class.from_connection_string.return_value = mock_client
    
    mock_database = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_database
    
    # Mock containers - first (model_versions) fails
    mock_database.create_container_if_not_exists.side_effect = Exception("Model versions container error")
    
    with pytest.raises(Exception, match="Model versions container error"):
        cosmos_client_module.get_cosmos_client("conn_str", "test_db")


@patch.object(cosmos_client_module, 'CosmosClient')
def test_get_cosmos_client_connection_error(mock_cosmos_client_class):
    """Test error when creating Cosmos client connection"""
    mock_cosmos_client_class.from_connection_string.side_effect = Exception("Connection error")
    
    with pytest.raises(Exception, match="Connection error"):
        cosmos_client_module.get_cosmos_client("conn_str", "test_db")


def test_get_next_version_first_version():
    """Test getting next version when no versions exist"""
    mock_container = MagicMock()
    mock_container.query_items.return_value = []
    
    result = cosmos_client_module.get_next_version(mock_container, "PETR4")
    
    assert result == "v1"
    mock_container.query_items.assert_called_once()


def test_get_next_version_existing_versions():
    """Test getting next version when versions exist"""
    mock_container = MagicMock()
    mock_container.query_items.return_value = [
        {"version": "v3"},
        {"version": "v2"},
        {"version": "v1"}
    ]
    
    result = cosmos_client_module.get_next_version(mock_container, "PETR4")
    
    assert result == "v4"


def test_get_next_version_query_error():
    """Test fallback to v1 when query fails"""
    mock_container = MagicMock()
    mock_container.query_items.side_effect = Exception("Query error")
    
    result = cosmos_client_module.get_next_version(mock_container, "PETR4")
    
    assert result == "v1"


def test_get_latest_version_no_versions():
    """Test getting latest version when no completed versions exist"""
    mock_container = MagicMock()
    mock_container.query_items.return_value = []
    
    result = cosmos_client_module.get_latest_version(mock_container, "PETR4")
    
    assert result is None


def test_get_latest_version_existing():
    """Test getting latest version when versions exist"""
    mock_container = MagicMock()
    mock_container.query_items.return_value = [
        {"version": "v3", "timestamp": "2024-01-03"},
        {"version": "v2", "timestamp": "2024-01-02"}
    ]
    
    result = cosmos_client_module.get_latest_version(mock_container, "PETR4")
    
    assert result == "v3"


def test_get_latest_version_query_error():
    """Test returning None when query fails"""
    mock_container = MagicMock()
    mock_container.query_items.side_effect = Exception("Query error")
    
    result = cosmos_client_module.get_latest_version(mock_container, "PETR4")
    
    assert result is None


@patch.object(cosmos_client_module, 'datetime')
def test_save_model_version_success(mock_datetime):
    """Test successful saving of model version"""
    mock_datetime.now.return_value = MagicMock()
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
    
    mock_container = MagicMock()
    
    metrics = {"accuracy": 0.95}
    hyperparams = {"lr": 0.001}
    
    cosmos_client_module.save_model_version(
        mock_container, "PETR4", "v1", metrics, hyperparams,
        "model.pkl", "scaler.pkl", "metrics.json", "completed"
    )
    
    # Check that upsert_item was called
    mock_container.upsert_item.assert_called_once()
    call_args = mock_container.upsert_item.call_args[0][0]
    assert call_args["id"] == "PETR4_v1"
    assert call_args["ticker"] == "PETR4"
    assert call_args["version"] == "v1"
    assert call_args["metrics"] == metrics
    assert call_args["hyperparams"] == hyperparams


@patch.object(cosmos_client_module, 'datetime')
def test_save_prediction_success(mock_datetime):
    """Test successful saving of prediction"""
    mock_datetime.now.return_value = MagicMock()
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
    
    mock_container = MagicMock()
    
    cosmos_client_module.save_prediction(
        mock_container, "PETR4", "v1", "2024-01-02", 25.5, "2024-01-01"
    )
    
    mock_container.upsert_item.assert_called_once()
    call_args = mock_container.upsert_item.call_args[0][0]
    assert call_args["id"] == "PETR4_v1_2024-01-02"
    assert call_args["predicted_price"] == 25.5


def test_get_prediction_found():
    """Test getting prediction when it exists"""
    mock_container = MagicMock()
    mock_prediction = {"id": "PETR4_v1_2024-01-02", "predicted_price": 25.5}
    mock_container.read_item.return_value = mock_prediction
    
    result = cosmos_client_module.get_prediction(mock_container, "PETR4", "v1", "2024-01-02")
    
    assert result == mock_prediction


def test_get_prediction_not_found():
    """Test getting prediction when it doesn't exist"""
    mock_container = MagicMock()
    
    def raise_not_found(*args, **kwargs):
        raise cosmos_client_module.exceptions.CosmosResourceNotFoundError()
    
    mock_container.read_item.side_effect = raise_not_found
    
    result = cosmos_client_module.get_prediction(mock_container, "PETR4", "v1", "2024-01-02")
    
    assert result is None


def test_get_prediction_error():
    """Test error when getting prediction"""
    mock_container = MagicMock()
    mock_container.read_item.side_effect = Exception("Read error")
    
    with pytest.raises(Exception, match="Read error"):
        cosmos_client_module.get_prediction(mock_container, "PETR4", "v1", "2024-01-02")