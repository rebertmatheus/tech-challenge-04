import sys
from pathlib import Path
import pytest
import importlib.util

# Load the config module dynamically
config_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "config.py"
spec = importlib.util.spec_from_file_location("config", str(config_path))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config


def test_get_tickers_success(monkeypatch):
    monkeypatch.setenv("TICKERS", "A, B, C")
    assert Config.get_tickers() == ["A", "B", "C"]


def test_get_tickers_with_spaces(monkeypatch):
    monkeypatch.setenv("TICKERS", " A , B , C ")
    assert Config.get_tickers() == ["A", "B", "C"]


def test_get_tickers_empty_values(monkeypatch):
    monkeypatch.setenv("TICKERS", "A,,B,")
    assert Config.get_tickers() == ["A", "B"]


def test_get_tickers_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TICKERS", raising=False)
    with pytest.raises(ValueError):
        Config.get_tickers()


def test_get_storage_config_defaults(monkeypatch):
    monkeypatch.setenv("AzureWebJobsStorage", "connstr")
    monkeypatch.delenv("BLOB_CONTAINER", raising=False)
    cfg = Config.get_storage_config()
    assert cfg["conn_str"] == "connstr"
    assert cfg["container"] == "techchallenge04storage"


def test_get_storage_config_custom(monkeypatch):
    monkeypatch.setenv("AzureWebJobsStorage", "cs")
    monkeypatch.setenv("BLOB_CONTAINER", "mycontainer")
    cfg = Config.get_storage_config()
    assert cfg["conn_str"] == "cs"
    assert cfg["container"] == "mycontainer"


def test_get_cosmos_config_defaults(monkeypatch):
    monkeypatch.setenv("COSMOS_DB_CONNECTION_STRING", "conn")
    monkeypatch.delenv("COSMOS_DB_DATABASE", raising=False)
    monkeypatch.delenv("COSMOS_DB_CONTAINER_MODEL_VERSIONS", raising=False)
    monkeypatch.delenv("COSMOS_DB_CONTAINER_TRAINING_METRICS", raising=False)
    cfg = Config.get_cosmos_config()
    assert cfg["conn_str"] == "conn"
    assert cfg["database"] == "techchallenge04"
    assert cfg["container_model_versions"] == "model_versions"
    assert cfg["container_training_metrics"] == "training_metrics"


def test_get_cosmos_config_custom(monkeypatch):
    monkeypatch.setenv("COSMOS_DB_CONNECTION_STRING", "cs")
    monkeypatch.setenv("COSMOS_DB_DATABASE", "mydb")
    monkeypatch.setenv("COSMOS_DB_CONTAINER_MODEL_VERSIONS", "mv")
    monkeypatch.setenv("COSMOS_DB_CONTAINER_TRAINING_METRICS", "tm")
    cfg = Config.get_cosmos_config()
    assert cfg["conn_str"] == "cs"
    assert cfg["database"] == "mydb"
    assert cfg["container_model_versions"] == "mv"
    assert cfg["container_training_metrics"] == "tm"


def test_get_model_config():
    cfg = Config.get_model_config()
    assert cfg["default_version"] == "v1"
    assert cfg["hyperparameters_path"] == "hyperparameters"
    assert cfg["models_path"] == "models"
    assert cfg["history_path"] == "history"


def test_enable_progress_bar_true_explicit(monkeypatch):
    monkeypatch.setenv("ENABLE_PROGRESS_BAR", "true")
    assert Config.enable_progress_bar() is True


def test_enable_progress_bar_false_explicit(monkeypatch):
    monkeypatch.setenv("ENABLE_PROGRESS_BAR", "false")
    assert Config.enable_progress_bar() is False


def test_enable_progress_bar_local(monkeypatch):
    monkeypatch.delenv("ENABLE_PROGRESS_BAR", raising=False)
    monkeypatch.delenv("WEBSITE_INSTANCE_ID", raising=False)
    assert Config.enable_progress_bar() is True


def test_enable_progress_bar_azure(monkeypatch):
    monkeypatch.delenv("ENABLE_PROGRESS_BAR", raising=False)
    monkeypatch.setenv("WEBSITE_INSTANCE_ID", "instance")
    assert Config.enable_progress_bar() is False