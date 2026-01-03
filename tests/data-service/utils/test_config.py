import sys
from pathlib import Path
import pytest
import importlib.util

# Load the config module dynamically
config_path = Path(__file__).resolve().parents[3] / "data-service" / "utils" / "config.py"
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


def test_get_date_range_defaults(monkeypatch):
    monkeypatch.delenv("START_DATE", raising=False)
    monkeypatch.delenv("END_DATE", raising=False)
    dr = Config.get_date_range()
    assert dr["start"] == "2018-01-01"
    assert dr["end"] == "2025-10-31"


def test_get_date_range_custom(monkeypatch):
    monkeypatch.setenv("START_DATE", "2020-01-01")
    monkeypatch.setenv("END_DATE", "2021-12-31")
    dr = Config.get_date_range()
    assert dr["start"] == "2020-01-01"
    assert dr["end"] == "2021-12-31"


def test_get_loopback_period_default(monkeypatch):
    monkeypatch.delenv("LOOPBACK_PERIOD", raising=False)
    assert Config.get_loopback_period() == 120


def test_get_loopback_period_custom_int(monkeypatch):
    monkeypatch.setenv("LOOPBACK_PERIOD", "300")
    assert Config.get_loopback_period() == 300


def test_get_loopback_period_invalid(monkeypatch):
    monkeypatch.setenv("LOOPBACK_PERIOD", "not-an-int")
    assert Config.get_loopback_period() == 120