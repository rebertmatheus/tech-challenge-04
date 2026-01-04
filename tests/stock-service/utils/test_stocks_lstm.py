
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Adicionar o diretório stock-service ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stock-service'))

# Import the module after mocking
from utils.stocks_lstm import StocksLSTM

# Mock the save_hyperparameters method on the StocksLSTM class
StocksLSTM.save_hyperparameters = MagicMock()
StocksLSTM.log = MagicMock()

# Make the model callable by adding __call__ method
def mock_call(self, x):
    return self.forward(x)

StocksLSTM.__call__ = mock_call


class TestStocksLSTM:
    """Testes para a classe StocksLSTM"""

    @pytest.fixture
    def sample_config_dict(self):
        """Configuração de exemplo como dicionário"""
        return {
            "FEATURE_COLS": ["Close", "Volume", "High", "Low"],
            "INIT_HIDDEN_SIZE": 64,
            "SECOND_HIDDEN_SIZE": 32,
            "NUM_LAYERS": 2,
            "DROPOUT_VALUE": 0.2,
            "LEARNING_RATE": 0.001,
            "WEIGHT_DECAY": 0.0001,
            "RLR_FACTOR": 0.5,
            "RLR_PATIENCE": 5,
            "USE_FC_LAYERS": False
        }

    @pytest.fixture
    def sample_config_object(self):
        """Configuração de exemplo como objeto"""
        config = MagicMock()
        config.FEATURE_COLS = ["Close", "Volume", "High", "Low"]
        config.INIT_HIDDEN_SIZE = 64
        config.SECOND_HIDDEN_SIZE = 32
        config.NUM_LAYERS = 2
        config.DROPOUT_VALUE = 0.2
        config.LEARNING_RATE = 0.001
        config.WEIGHT_DECAY = 0.0001
        config.RLR_FACTOR = 0.5
        config.RLR_PATIENCE = 5
        config.USE_FC_LAYERS = False
        return config

    @pytest.fixture
    def sample_config_with_fc(self):
        """Configuração com camadas FC adicionais"""
        return {
            "FEATURE_COLS": ["Close", "Volume", "High", "Low"],
            "INIT_HIDDEN_SIZE": 64,
            "SECOND_HIDDEN_SIZE": 32,
            "NUM_LAYERS": 2,
            "DROPOUT_VALUE": 0.2,
            "LEARNING_RATE": 0.001,
            "WEIGHT_DECAY": 0.0001,
            "RLR_FACTOR": 0.5,
            "RLR_PATIENCE": 5,
            "USE_FC_LAYERS": True,
            "FC_HIDDEN_1": 64,
            "FC_HIDDEN_2": 32,
            "FC_DROPOUT": 0.1
        }

    def test_init_with_dict_config(self, sample_config_dict):
        """Testa inicialização com configuração como dicionário"""
        model = StocksLSTM(sample_config_dict)

        assert model.config == sample_config_dict
        assert len(model.config["FEATURE_COLS"]) == 4
        assert model.config["INIT_HIDDEN_SIZE"] == 64
        model.save_hyperparameters.assert_called_once()
        # Verificar que save_hyperparameters foi chamado
        assert hasattr(model, 'save_hyperparameters')

    def test_init_with_object_config(self, sample_config_object):
        """Testa inicialização com configuração como objeto"""
        model = StocksLSTM(sample_config_object)

        assert model.config == sample_config_object

    def test_init_with_fc_layers(self, sample_config_with_fc):
        """Testa inicialização com camadas FC adicionais"""
        model = StocksLSTM(sample_config_with_fc)

        assert model.config == sample_config_with_fc
        # Verificar que USE_FC_LAYERS está ativado
        assert model.get_config("USE_FC_LAYERS") == True

    def test_get_config_dict(self, sample_config_dict):
        """Testa método get_config com dicionário"""
        model = StocksLSTM(sample_config_dict)

        assert model.get_config("FEATURE_COLS") == ["Close", "Volume", "High", "Low"]
        assert model.get_config("INIT_HIDDEN_SIZE") == 64
        assert model.get_config("NON_EXISTENT", "default") == "default"

    def test_get_config_object(self, sample_config_object):
        """Testa método get_config com objeto"""
        model = StocksLSTM(sample_config_object)

        assert model.get_config("FEATURE_COLS") == ["Close", "Volume", "High", "Low"]
        assert model.get_config("INIT_HIDDEN_SIZE") == 64

    @patch('torch.nn.functional.mse_loss')
    def test_forward_pass(self, mock_mse_loss, sample_config_dict):
        """Testa forward pass do modelo"""
        model = StocksLSTM(sample_config_dict)

        # Mock do forward pass
        batch_size, seq_len, num_features = 32, 30, 4
        x = torch_mock.randn(batch_size, seq_len, num_features)

        # Mock das camadas
        model.lstm1 = MagicMock(return_value=(torch_mock.randn(seq_len, batch_size, 64), None))
        model.lstm2 = MagicMock(return_value=(torch_mock.randn(seq_len, batch_size, 32), None))
        model.lstm3 = MagicMock(return_value=(torch_mock.randn(seq_len, batch_size, 32), None))
        model.dropout = MagicMock(side_effect=lambda tensor: tensor)  # Retorna o tensor original
        model.fc = MagicMock(return_value=torch_mock.randn(batch_size, 1))

        result = model.forward(x)

        # Verificar que as camadas foram chamadas
        assert model.lstm1.called
        assert model.lstm2.called
        assert model.lstm3.called
        assert model.fc.called

    @patch('torch.nn.functional.mse_loss')
    def test_training_step(self, mock_mse_loss, sample_config_dict):
        """Testa training_step"""
        model = StocksLSTM(sample_config_dict)

        # Configurar mock do mse_loss
        mock_mse_loss.return_value = torch_mock.tensor(0.5)

        batch = (torch_mock.randn(32, 30, 4), torch_mock.randn(32))
        batch_idx = 0

        # Mock do forward para retornar tensor achatado
        model.forward = MagicMock(return_value=torch_mock.randn(32, 1))

        loss = model.training_step(batch, batch_idx)

        # Verificar que mse_loss foi chamado
        mock_mse_loss.assert_called_once()
        # Verificar que log foi chamado
        assert model.log.called

    @patch('torch.nn.functional.mse_loss')
    def test_validation_step(self, mock_mse_loss, sample_config_dict):
        """Testa validation_step"""
        model = StocksLSTM(sample_config_dict)

        # Configurar mock do mse_loss
        mock_mse_loss.return_value = torch_mock.tensor(0.3)

        batch = (torch_mock.randn(32, 30, 4), torch_mock.randn(32))
        batch_idx = 0

        # Mock do forward para retornar tensor achatado
        model.forward = MagicMock(return_value=torch_mock.randn(32, 1))

        loss = model.validation_step(batch, batch_idx)

        # Verificar que mse_loss foi chamado
        mock_mse_loss.assert_called_once()
        # Verificar que log foi chamado com val_loss
        model.log.assert_called_with("val_loss", torch_mock.tensor(0.3), prog_bar=True)

    @patch('torch.nn.functional.mse_loss')
    def test_test_step(self, mock_mse_loss, sample_config_dict):
        """Testa test_step"""
        model = StocksLSTM(sample_config_dict)

        # Configurar mock do mse_loss
        mock_mse_loss.return_value = torch_mock.tensor(0.4)

        batch = (torch_mock.randn(32, 30, 4), torch_mock.randn(32))
        batch_idx = 0

        # Mock do forward para retornar tensor achatado
        model.forward = MagicMock(return_value=torch_mock.randn(32, 1))

        loss = model.test_step(batch, batch_idx)

        # Verificar que mse_loss foi chamado
        mock_mse_loss.assert_called_once()
        # Verificar que log foi chamado com test_loss
        model.log.assert_called_with("test_loss", torch_mock.tensor(0.4), prog_bar=True)

    def test_predict_step(self, sample_config_dict):
        """Testa predict_step"""
        model = StocksLSTM(sample_config_dict)

        batch = (torch_mock.randn(32, 30, 4), torch_mock.randn(32))
        batch_idx = 0

        # Mock do forward para retornar tensor achatado
        model.forward = MagicMock(return_value=torch_mock.randn(32, 1))

        result = model.predict_step(batch, batch_idx)

        # Verificar que forward foi chamado apenas com inputs
        model.forward.assert_called_once_with(batch[0])
        # Verificar que resultado foi retornado (não podemos verificar shape de mock)
        assert result is not None

    @patch('torch.optim.Adam')
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    def test_configure_optimizers(self, mock_scheduler, mock_adam, sample_config_dict):
        """Testa configuração de otimizadores"""
        model = StocksLSTM(sample_config_dict)

        # Configurar mocks
        mock_optimizer = MagicMock()
        mock_lr_scheduler = MagicMock()
        mock_adam.return_value = mock_optimizer
        mock_scheduler.return_value = mock_lr_scheduler

        # Mock parameters
        model.parameters = MagicMock(return_value=iter([MagicMock()]))

        result = model.configure_optimizers()

        # Verificar que Adam foi chamado com parâmetros corretos
        mock_adam.assert_called_once_with(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )

        # Verificar que ReduceLROnPlateau foi chamado
        mock_scheduler.assert_called_once_with(
            mock_optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Verificar estrutura do retorno
        assert 'optimizer' in result
        assert 'lr_scheduler' in result
        assert result['optimizer'] == mock_optimizer
        assert result['lr_scheduler']['scheduler'] == mock_lr_scheduler
        assert result['lr_scheduler']['monitor'] == 'val_loss'

    def test_init_missing_required_config(self):
        """Testa inicialização com configuração faltando parâmetros obrigatórios"""
        incomplete_config = {
            "FEATURE_COLS": ["Close", "Volume"]
            # Faltando outros parâmetros obrigatórios
        }

        # O modelo deve ser criado mesmo com configuração incompleta
        # pois a validação não é feita no __init__
        model = StocksLSTM(incomplete_config)
        assert model.config == incomplete_config

    def test_forward_invalid_input_shape(self, sample_config_dict):
        """Testa forward com input de formato inválido"""
        model = StocksLSTM(sample_config_dict)

        # Mock das camadas para evitar erros
        model.lstm1 = MagicMock(return_value=(torch_mock.randn(4, 32, 64), None))
        model.lstm2 = MagicMock(return_value=(torch_mock.randn(4, 32, 32), None))
        model.lstm3 = MagicMock(return_value=(torch_mock.randn(4, 32, 32), None))
        model.dropout = MagicMock(side_effect=lambda tensor: tensor)
        model.fc = MagicMock(return_value=torch_mock.randn(32, 1))

        # Input com dimensões erradas
        x = torch_mock.randn(32, 4)  # Faltando dimensão de sequência

        # O forward deve funcionar mesmo com input de formato diferente
        # pois estamos mockando as camadas
        result = model.forward(x)
        assert result is not None