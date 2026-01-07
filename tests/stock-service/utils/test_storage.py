import io
import json
import datetime
from unittest.mock import MagicMock
import pandas as pd
from pathlib import Path
import pytest
from azure.core.exceptions import ResourceExistsError

import importlib.util

# Load the storage module dynamically
storage_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "storage.py"
spec = importlib.util.spec_from_file_location("storage", str(storage_path))
storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage_module)
get_storage_client = storage_module.get_storage_client


@pytest.fixture
def container_client_mock():
    container = MagicMock()
    # get_blob_client retorna um MagicMock por padrão; cada teste pode configurar retornos específicos
    container.get_blob_client.return_value = MagicMock()
    return container

def test_get_storage_client_creates_container(monkeypatch):
    container_client = MagicMock()
    service = MagicMock()
    service.get_container_client.return_value = container_client
    monkeypatch.setattr(storage_module.BlobServiceClient, "from_connection_string", lambda conn: service)
    returned = storage_module.get_storage_client("fake-conn", "meu-container")
    assert returned is container_client
    container_client.create_container.assert_called_once()

def test_get_storage_client_handles_existing_container(monkeypatch):
    container_client = MagicMock()
    container_client.create_container.side_effect = ResourceExistsError("exists")
    service = MagicMock()
    service.get_container_client.return_value = container_client
    monkeypatch.setattr(storage_module.BlobServiceClient, "from_connection_string", lambda conn: service)
    returned = storage_module.get_storage_client("fake-conn", "meu-container")
    assert returned is container_client
    container_client.create_container.assert_called_once()

def test_load_hyperparameters_success(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b'{"lr": 0.01, "batch": 32}'
    container_client_mock.get_blob_client.return_value = blob_client
    out = storage_module.load_hyperparameters(container_client_mock, "PETR4")
    assert isinstance(out, dict)
    assert out["lr"] == 0.01
    assert out["batch"] == 32

def test_load_hyperparameters_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_hyperparameters(container_client_mock, "PETR4")

def test_load_history_data_success(monkeypatch, container_client_mock):
    df_expected = pd.DataFrame({"col": [1, 2, 3]})
    monkeypatch.setattr(pd, "read_parquet", lambda buf: df_expected)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"irrelevant-bytes"
    container_client_mock.get_blob_client.return_value = blob_client
    df = storage_module.load_history_data(container_client_mock, "PETR4")
    pd.testing.assert_frame_equal(df, df_expected)

def test_load_history_data_empty(monkeypatch, container_client_mock):
    df_empty = pd.DataFrame()
    monkeypatch.setattr(pd, "read_parquet", lambda buf: df_empty)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"irrelevant"
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(ValueError):
        storage_module.load_history_data(container_client_mock, "PETR4")

def test_load_history_data_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_history_data(container_client_mock, "PETR4")

def test_save_model_uploads_blobs(container_client_mock):
    model_bytes = b"checkpoint-bytes"
    scaler_bytes = b"scaler-bytes"
    metrics_bytes = b"metrics-bytes"
    blob_client = MagicMock()
    container_client_mock.get_blob_client.return_value = blob_client
    result = storage_module.save_model(container_client_mock, "PETR4", "v1", model_bytes, scaler_bytes, metrics_bytes)
    upload_calls = blob_client.upload_blob.call_count
    assert upload_calls >= 3
    assert result["model_path"] == "models/PETR4_v1.ckpt"
    assert result["scaler_path"] == "models/PETR4_v1_scaler.pkl"
    assert "metrics_path" in result and result["metrics_path"] == "models/PETR4_v1_metrics.pkl"

def test_load_metrics_success(monkeypatch, container_client_mock):
    fake_metrics = {"val_loss": 0.1}
    monkeypatch.setattr(storage_module.joblib, "load", lambda buf: fake_metrics)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"some-bytes"
    container_client_mock.get_blob_client.return_value = blob_client
    out = storage_module.load_metrics(container_client_mock, "PETR4", "v1")
    assert out == fake_metrics

def test_load_metrics_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_metrics(container_client_mock, "PETR4", "v1")

def test_load_model_success(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"checkpoint-bytes"
    container_client_mock.get_blob_client.return_value = blob_client
    out = storage_module.load_model(container_client_mock, "PETR4", "v1")
    assert out == b"checkpoint-bytes"

def test_load_model_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_model(container_client_mock, "PETR4", "v1")

def test_load_scaler_success(monkeypatch, container_client_mock):
    fake_scaler = {"feature_scaler": "f", "target_scaler": "t"}
    monkeypatch.setattr(storage_module.joblib, "load", lambda buf: fake_scaler)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"scaler-bytes"
    container_client_mock.get_blob_client.return_value = blob_client
    out = storage_module.load_scaler(container_client_mock, "PETR4", "v1")
    assert out == fake_scaler

def test_load_scaler_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_scaler(container_client_mock, "PETR4", "v1")

def test_load_daily_data_success_string_date(monkeypatch, container_client_mock):
    df_expected = pd.DataFrame({"x": [10]})
    monkeypatch.setattr(pd, "read_parquet", lambda buf: df_expected)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"irrelevant"
    container_client_mock.get_blob_client.return_value = blob_client
    df = storage_module.load_daily_data(container_client_mock, "PETR4", "2024-12-31")
    pd.testing.assert_frame_equal(df, df_expected)

def test_load_daily_data_not_found(container_client_mock):
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(FileNotFoundError):
        storage_module.load_daily_data(container_client_mock, "PETR4", "2024-12-31")

def test_load_daily_data_empty(monkeypatch, container_client_mock):
    df_empty = pd.DataFrame()
    monkeypatch.setattr(pd, "read_parquet", lambda buf: df_empty)
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    blob_client.download_blob.return_value.readall.return_value = b"irrelevant"
    container_client_mock.get_blob_client.return_value = blob_client
    with pytest.raises(ValueError):
        storage_module.load_daily_data(container_client_mock, "PETR4", "2024-12-31")