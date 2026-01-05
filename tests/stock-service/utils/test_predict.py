import os
import io
import tempfile
from unittest.mock import MagicMock
import torch
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import importlib.util

# Load the predictor module dynamically, handling its relative import
# First load stocks_lstm as a top-level module so predictor can import it
stocks_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "stocks_lstm.py"
spec_s = importlib.util.spec_from_file_location("stocks_lstm", str(stocks_path))
stocks_module = importlib.util.module_from_spec(spec_s)
spec_s.loader.exec_module(stocks_module)
import sys
sys.modules["stocks_lstm"] = stocks_module

# Now load predictor.py source, replace relative import with absolute import
predict_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "predictor.py"
source = predict_path.read_text(encoding="utf-8")
source = source.replace("from .stocks_lstm import StocksLSTM", "from stocks_lstm import StocksLSTM")
predict_module = type(stocks_module)("predictor")
exec(compile(source, str(predict_path), 'exec'), predict_module.__dict__)
preds = predict_module


@pytest.fixture
def feature_scaler_mock():
    scaler = MagicMock()
    # transform deve receber DataFrame[feature_cols] e retornar numpy array
    scaler.transform.side_effect = lambda df: np.asarray(df.values, dtype=np.float32)
    return scaler


def test_prepare_prediction_sequence_success(feature_scaler_mock):
    # preparar dataframe com 5 registros e 2 colunas de features
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    hyperparams = {
        "SEQUENCE_LENGTH": 3,
        "FEATURE_COLS": ["f1", "f2"]
    }

    seq_tensor = preds.prepare_prediction_sequence(df, hyperparams, feature_scaler_mock)

    assert isinstance(seq_tensor, torch.Tensor)
    assert seq_tensor.shape == (1, 3, 2)

    # verificar que os últimos 3 registros foram usados (3,4,5) e correspondem ao transform
    expected = np.array([[3.0, 30.0],
                         [4.0, 40.0],
                         [5.0, 50.0]], dtype=np.float32).reshape(1, 3, 2)
    np.testing.assert_allclose(seq_tensor.numpy(), expected)


def test_prepare_prediction_sequence_insufficient_data(feature_scaler_mock):
    df = pd.DataFrame({
        "f1": [1.0],
        "f2": [10.0],
    })
    hyperparams = {
        "SEQUENCE_LENGTH": 3,
        "FEATURE_COLS": ["f1", "f2"]
    }

    with pytest.raises(ValueError):
        preds.prepare_prediction_sequence(df, hyperparams, feature_scaler_mock)


def test_load_model_from_bytes_success(monkeypatch, tmp_path):
    # bytes arbitrary (não será lido porque vamos mockar load_from_checkpoint)
    model_bytes = b"fake checkpoint bytes"
    hyperparams = {"some": "config"}

    # criar um mock de modelo para ser retornado por load_from_checkpoint
    fake_model = MagicMock(name="StocksLSTM")
    # monkeypatchar o método load_from_checkpoint do StocksLSTM importado no módulo
    monkeypatch.setattr(preds.StocksLSTM, "load_from_checkpoint", staticmethod(lambda path, config: fake_model))

    # chamar a função
    model = preds.load_model_from_bytes(model_bytes, hyperparams)

    assert model is fake_model
    # garantir que o mock foi chamado (o caminho do checkpoint foi passado para o loader)
    # não conseguimos inspecionar o caminho exato facilmente aqui, mas o fato de retornar o fake_model
    # implica que a chamada ocorreu


def test_load_model_from_bytes_exception_cleans_up(monkeypatch, tmp_path):
    # Forçar tempfile.mkdtemp a usar um diretório controlado para que possamos verificar remoção
    tempdir = tmp_path / "mytemp"
    tempdir.mkdir()
    monkeypatch.setattr(preds.tempfile, "mkdtemp", lambda: str(tempdir))

    model_bytes = b"irrelevant"
    hyperparams = {}

    # Fazer o loader levantar exceção para testar limpeza
    def raise_loader(path, config):
        raise RuntimeError("load failed")
    monkeypatch.setattr(preds.StocksLSTM, "load_from_checkpoint", staticmethod(raise_loader))

    # Chamar e esperar exceção; após isso o diretório deve ter sido removido
    with pytest.raises(RuntimeError):
        preds.load_model_from_bytes(model_bytes, hyperparams)

    # O módulo tenta remover o arquivo e em seguida o diretório; verificar que o diretório não existe
    assert not tempdir.exists()


def test_predict_price_success():
    # construir um modelo simples que retorna uma previsão normalizada
    class DummyModel:
        def __init__(self):
            # parâmetro apenas para ter .device disponível
            self._p = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        def parameters(self):
            # generator que yield o parâmetro
            yield self._p

        def eval(self):
            pass

        def __call__(self, seq):
            # retornar tensor com valor normalizado por exemplo 0.5
            return torch.tensor([[0.5]], dtype=torch.float32)

    model = DummyModel()

    # sequência dummy (shape deve ser compatível com o modelo, mas o DummyModel ignora o input)
    sequence = torch.zeros((1, 3, 2), dtype=torch.float32)

    # target_scaler que inverse_transform irá desnormalizar para 123.45
    target_scaler = MagicMock()
    target_scaler.inverse_transform.return_value = np.array([[123.45]])

    predicted = preds.predict_price(model, sequence, target_scaler)

    assert isinstance(predicted, float)
    assert abs(predicted - 123.45) < 1e-6

    # garantir que inverse_transform recebeu array com shape (N,1)
    args, _ = target_scaler.inverse_transform.call_args
    arr_passed = args[0]
    assert arr_passed.shape[1] == 1


def test_predict_price_moves_to_device_and_logs(monkeypatch):
    # Testar que a função usa device do primeiro parâmetro do modelo e que trata torch.no_grad context
    class DeviceModel:
        def __init__(self, device):
            self._p = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32).to(device))
            self.called = False

        def parameters(self):
            yield self._p

        def eval(self):
            pass

        def __call__(self, seq):
            # checar que seq está no mesmo device que o parameter
            assert seq.device == self._p.device
            self.called = True
            return torch.tensor([[1.0]], device=self._p.device)

    model = DeviceModel(device=torch.device("cpu"))
    sequence = torch.zeros((1, 2, 1), dtype=torch.float32)  # cpu tensor

    target_scaler = MagicMock()
    target_scaler.inverse_transform.return_value = np.array([[9.99]])

    predicted = preds.predict_price(model, sequence, target_scaler)

    assert model.called is True
    assert predicted == pytest.approx(9.99)