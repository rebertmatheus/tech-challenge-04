import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Adicionar o diretório stock-service ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stock-service'))

from utils.predictor import prepare_prediction_sequence, load_model_from_bytes, predict_price


class TestPredictor:
    """Testes para o módulo predictor.py"""

    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para testes"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        data = {
            'Close': np.random.uniform(10, 50, 50),
            'Volume': np.random.uniform(1000, 10000, 50),
            'High': np.random.uniform(15, 55, 50),
            'Low': np.random.uniform(5, 45, 50)
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def hyperparams(self):
        """Hiperparâmetros de exemplo"""
        return {
            "SEQUENCE_LENGTH": 30,
            "FEATURE_COLS": ["Close", "Volume", "High", "Low"],
            "TARGET_COL": "Close"
        }

    @pytest.fixture
    def mock_scaler(self):
        """Mock scaler para testes"""
        scaler = MagicMock()
        scaler.transform.return_value = np.random.randn(30, 4)
        return scaler

    def test_prepare_prediction_sequence_success(self, sample_data, hyperparams, mock_scaler):
        """Testa preparação de sequência com dados suficientes"""
        result = prepare_prediction_sequence(sample_data, hyperparams, mock_scaler)

        # With mocked torch, result will be a MagicMock
        assert result is not None
        # Verify that torch.tensor was called
        torch.tensor.assert_called()

        # Verificar que scaler.transform foi chamado
        mock_scaler.transform.assert_called_once()

    def test_prepare_prediction_sequence_insufficient_data(self, sample_data, hyperparams, mock_scaler):
        """Testa erro quando não há dados suficientes"""
        # DataFrame com menos registros que SEQUENCE_LENGTH
        small_df = sample_data.head(20)

        with pytest.raises(ValueError, match="Dados insuficientes"):
            prepare_prediction_sequence(small_df, hyperparams, mock_scaler)

    def test_prepare_prediction_sequence_exact_data(self, sample_data, hyperparams, mock_scaler):
        """Testa com exatamente SEQUENCE_LENGTH registros"""
        exact_df = sample_data.head(30)

        result = prepare_prediction_sequence(exact_df, hyperparams, mock_scaler)

        # With mocked torch, result will be a MagicMock
        assert result is not None
        # Verify that torch.tensor was called
        torch.tensor.assert_called()

    @patch('utils.predictor.logger')
    def test_prepare_prediction_sequence_logging(self, mock_logger, sample_data, hyperparams, mock_scaler):
        """Testa logging na preparação de sequência"""
        prepare_prediction_sequence(sample_data, hyperparams, mock_scaler)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Sequência preparada" in call_args
        assert "shape" in call_args

    @patch('utils.predictor.logger')
    def test_prepare_prediction_sequence_exception_logging(self, mock_logger, sample_data, hyperparams, mock_scaler):
        """Testa logging de exceções"""
        mock_scaler.transform.side_effect = Exception("Transform error")

        with pytest.raises(Exception):
            prepare_prediction_sequence(sample_data, hyperparams, mock_scaler)

        mock_logger.exception.assert_called_once_with("Erro ao preparar sequência para predição")

    @patch('utils.predictor.StocksLSTM.load_from_checkpoint')
    @patch('utils.predictor.tempfile.mkdtemp')
    @patch('utils.predictor.os.path.join')
    @patch('utils.predictor.os.path.exists')
    @patch('utils.predictor.os.remove')
    @patch('utils.predictor.os.rmdir')
    @patch('builtins.open', new_callable=MagicMock)
    def test_load_model_from_bytes_success(self, mock_open, mock_rmdir, mock_remove, mock_exists,
                                          mock_join, mock_mkdtemp, mock_load_checkpoint, hyperparams):
        """Testa carregamento de modelo a partir de bytes"""
        # Configurar mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_join.return_value = "/tmp/test_dir/model.ckpt"
        mock_exists.return_value = True

        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model

        # Mock do context manager do open
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        model_bytes = b"fake_model_data"

        result = load_model_from_bytes(model_bytes, hyperparams)

        assert result == mock_model

        # Verificar que arquivo temporário foi criado e removido
        mock_mkdtemp.assert_called_once()
        mock_load_checkpoint.assert_called_once_with("/tmp/test_dir/model.ckpt", config=hyperparams)
        mock_remove.assert_called_once_with("/tmp/test_dir/model.ckpt")
        mock_rmdir.assert_called_once_with("/tmp/test_dir")

        # Verificar que os bytes foram escritos no arquivo
        mock_open.assert_called_once_with("/tmp/test_dir/model.ckpt", 'wb')
        mock_file.write.assert_called_once_with(model_bytes)

    @patch('utils.predictor.logger')
    @patch('utils.predictor.StocksLSTM.load_from_checkpoint')
    @patch('utils.predictor.tempfile.mkdtemp')
    def test_load_model_from_bytes_exception_logging(self, mock_mkdtemp, mock_load_checkpoint, mock_logger, hyperparams):
        """Testa logging de exceções no carregamento do modelo"""
        mock_mkdtemp.side_effect = Exception("Temp dir error")

        with pytest.raises(Exception):
            load_model_from_bytes(b"fake_data", hyperparams)

        mock_logger.exception.assert_called_once_with("Erro ao carregar modelo de bytes")

    @patch('utils.predictor.logger')
    def test_predict_price_success(self, mock_logger):
        """Testa execução de predição com sucesso"""
        # Criar mock do modelo
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters.return_value = iter([mock_param])  # Retornar iterador
        mock_model.return_value = torch.tensor([[0.5]])  # Predição normalizada

        # Mock scalers
        mock_target_scaler = MagicMock()
        mock_target_scaler.inverse_transform.return_value = np.array([[25.0]])

        # Tensor de entrada
        sequence = torch.randn(1, 30, 4)

        result = predict_price(mock_model, sequence, mock_target_scaler)

        assert isinstance(result, float)
        assert result == 25.0

        # Verificar chamadas
        mock_model.eval.assert_called_once()
        mock_target_scaler.inverse_transform.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch('utils.predictor.logger')
    def test_predict_price_exception_logging(self, mock_logger):
        """Testa logging de exceções na predição"""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.parameters.side_effect = Exception("Model error")

        with pytest.raises(Exception):
            predict_price(mock_model, torch.randn(1, 30, 4), MagicMock())

        mock_logger.exception.assert_called_once_with("Erro ao executar predição")

    def test_predict_price_numpy_conversion(self):
        """Testa conversão correta de tensores numpy"""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.return_value = torch.tensor([[0.8]])

        mock_scaler = MagicMock()
        mock_scaler.inverse_transform.return_value = np.array([[42.5]])

        result = predict_price(mock_model, torch.randn(1, 30, 4), mock_scaler)

        assert isinstance(result, float)
        assert result == 42.5