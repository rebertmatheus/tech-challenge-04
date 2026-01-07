import pytest
import torch
import numpy as np
from pathlib import Path
import importlib.util

# carregar module stocks_lstm dinamicamente (diretório contém hífen)
stocks_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "stocks_lstm.py"
spec = importlib.util.spec_from_file_location("stocks_lstm", str(stocks_path))
stocks_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stocks_module)
StocksLSTM = stocks_module.StocksLSTM


def make_config(**overrides):
    cfg = {
        "FEATURE_COLS": ["f1", "f2"],
        "INIT_HIDDEN_SIZE": 8,
        "SECOND_HIDDEN_SIZE": 8,
        "NUM_LAYERS": 1,
        "DROPOUT_VALUE": 0.0,
        "LEARNING_RATE": 1e-3,
        "WEIGHT_DECAY": 0.0,
        "RLR_FACTOR": 0.1,
        "RLR_PATIENCE": 2,
    }
    cfg.update(overrides)
    return cfg


def test_get_config_dict_and_object():
    cfg = make_config()
    m = StocksLSTM(cfg)
    assert m.get_config("LEARNING_RATE") == 1e-3

    class CfgObj:
        FEATURE_COLS = ["f1", "f2"]
        INIT_HIDDEN_SIZE = 4
        SECOND_HIDDEN_SIZE = 4
        NUM_LAYERS = 1
        DROPOUT_VALUE = 0.0
        LEARNING_RATE = 0.5
        WEIGHT_DECAY = 0.0
        RLR_FACTOR = 0.1
        RLR_PATIENCE = 2

    m2 = StocksLSTM(CfgObj)
    assert pytest.approx(m2.get_config("LEARNING_RATE"), rel=1e-6) == 0.5


def test_forward_shape_and_values():
    cfg = make_config()
    model = StocksLSTM(cfg)
    model.eval()

    batch_size = 2
    seq_len = 5
    num_features = len(cfg["FEATURE_COLS"])
    x = torch.randn(batch_size, seq_len, num_features, dtype=torch.float32)

    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, 1)


def test_configure_optimizers_returns_structure():
    cfg = make_config()
    model = StocksLSTM(cfg)
    opt_conf = model.configure_optimizers()
    assert isinstance(opt_conf, dict)
    assert "optimizer" in opt_conf
    assert "lr_scheduler" in opt_conf
    sch_conf = opt_conf["lr_scheduler"]
    assert "scheduler" in sch_conf and "monitor" in sch_conf


def test_training_step_computes_mse():
    cfg = make_config()
    model = StocksLSTM(cfg)
    model.eval()

    batch_size = 3
    seq_len = 4
    num_features = len(cfg["FEATURE_COLS"])

    inputs = torch.zeros(batch_size, seq_len, num_features, dtype=torch.float32)
    # set targets to zeros so mse with zero outputs should be small
    targets = torch.zeros(batch_size, dtype=torch.float32)

    loss = model.training_step((inputs, targets), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # manual mse check
    outputs = model(inputs).flatten()
    mse = torch.nn.functional.mse_loss(outputs, targets)
    assert pytest.approx(loss.item(), rel=1e-6) == mse.item()
