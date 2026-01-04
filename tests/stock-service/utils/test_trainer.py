import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import sys

# Mock torch before importing trainer
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()

# Mock pytorch_lightning and its submodules
pl_mock = MagicMock()
sys.modules['pytorch_lightning'] = pl_mock
sys.modules['pytorch_lightning.callbacks'] = MagicMock()
sys.modules['pytorch_lightning.loggers'] = MagicMock()
sys.modules['pytorch_lightning.loggers.CSVLogger'] = MagicMock()
sys.modules['pytorch_lightning.utilities'] = MagicMock()
sys.modules['pytorch_lightning.utilities.rank_zero'] = MagicMock()

# Mock other dependencies
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Adicionar o diretório stock-service ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stock-service'))

from utils.trainer import ModelTrainer


class TestModelTrainer:
    """Testes para a classe ModelTrainer"""

    @pytest.fixture
    def sample_dataframe(self):
        """DataFrame de exemplo para testes"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')  # Dados suficientes para treinamento
        np.random.seed(42)
        data = {
            'Close': np.random.uniform(10, 50, 200),
            'Volume': np.random.uniform(1000, 10000, 200),
            'High': np.random.uniform(15, 55, 200),
            'Low': np.random.uniform(5, 45, 200),
            'DROP_COL': np.random.uniform(0, 1, 200)  # Coluna a ser removida
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def hyperparams(self):
        """Hiperparâmetros de exemplo"""
        return {
            "SEQUENCE_LENGTH": 30,
            "FEATURE_COLS": ["Close", "Volume", "High", "Low"],
            "TARGET_COL": "Close",
            "DROP_COLUMNS": ["DROP_COL"],
            "BATCH_SIZE": 32,
            "EPOCHS": 10,
            "TRAIN_RATIO": 0.7,
            "VAL_RATIO": 0.2,
            "LEARNING_RATE": 0.001,
            "WEIGHT_DECAY": 0.0001,
            "GRADIENT_CLIP_VAL": 1.0,
            "LOG_EVERY_N_STEPS": 5,
            "ES_PATIENCE": 10,
            "ES_MIN_DELTA": 0.001,
            "RLR_FACTOR": 0.5,
            "RLR_PATIENCE": 5
        }

    @pytest.fixture
    def mock_model(self):
        """Mock do modelo StocksLSTM"""
        model = MagicMock()
        model.parameters.return_value = [MagicMock()]
        return model

    @pytest.fixture
    def trainer_instance(self):
        """Instância do ModelTrainer"""
        return ModelTrainer()

    def test_init(self, trainer_instance):
        """Testa inicialização do ModelTrainer"""
        assert trainer_instance.temp_dir is None

    @patch('utils.trainer.StocksLSTM')
    @patch('utils.trainer.SequenceDataset')
    @patch('utils.trainer.DataLoader')
    @patch('utils.trainer.pl.Trainer')
    @patch('utils.trainer.ModelTrainer._create_callbacks')
    @patch('utils.trainer.ModelTrainer._calculate_all_metrics')
    @patch('utils.trainer.ModelTrainer._serialize_model')
    @patch('utils.trainer.ModelTrainer._serialize_scalers')
    @patch('utils.trainer.ModelTrainer._serialize_metrics')
    @patch('utils.trainer.tempfile.mkdtemp')
    @patch('utils.trainer.os.path.exists')
    @patch('shutil.rmtree')
    @patch('utils.trainer.logger')
    def test_train_full_pipeline(self, mock_logger, mock_rmtree, mock_exists, mock_mkdtemp,
                                mock_serialize_metrics, mock_serialize_scalers, mock_serialize_model,
                                mock_calculate_metrics, mock_create_callbacks, mock_trainer_class,
                                mock_dataloader_class, mock_sequence_dataset_class, mock_stocks_lstm,
                                trainer_instance, sample_dataframe, hyperparams, mock_model):
        """Testa pipeline completo de treinamento"""
        # Configurar mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True

        # Mock datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        mock_feature_scaler = MagicMock()
        mock_target_scaler = MagicMock()

        # Configurar SequenceDataset para retornar datasets com scalers
        mock_sequence_dataset_class.side_effect = [mock_train_dataset, mock_val_dataset, mock_test_dataset]
        mock_train_dataset.get_scalers.return_value = (mock_feature_scaler, mock_target_scaler)
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_trainer_instance.fit.return_value = None

        # Mock callbacks
        mock_callbacks = [MagicMock(), MagicMock()]  # EarlyStopping e ModelCheckpoint
        mock_create_callbacks.return_value = mock_callbacks
        mock_callbacks[1].best_model_path = "/tmp/test_dir/best-model.ckpt"  # ModelCheckpoint

        # Configurar load_from_checkpoint no mock_model
        mock_stocks_lstm.load_from_checkpoint.return_value = mock_model

        # Mock métricas
        mock_metrics = {"validation": {"mae": 1.0}, "test": {"mae": 1.2}}
        mock_calculate_metrics.return_value = mock_metrics

        # Mock serialização
        mock_serialize_model.return_value = b"model_bytes"
        mock_serialize_scalers.return_value = b"scaler_bytes"
        mock_serialize_metrics.return_value = b"metrics_bytes"

        # Mock DataLoader
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_dataloader_class.side_effect = [mock_train_loader, mock_val_loader, mock_test_loader]

        # Executar treinamento
        result = trainer_instance.train(mock_model, "PETR4", hyperparams, sample_dataframe)

        # Verificar resultado
        model_bytes, scaler_bytes, metrics_bytes, metrics = result
        assert model_bytes == b"model_bytes"
        assert scaler_bytes == b"scaler_bytes"
        assert metrics_bytes == b"metrics_bytes"
        assert metrics == mock_metrics

        # Verificar que diretório temporário foi limpo
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    def test_prepare_data(self, trainer_instance, sample_dataframe, hyperparams):
        """Testa preparação de dados (remoção de colunas)"""
        result = trainer_instance._prepare_data(sample_dataframe, hyperparams)

        # Verificar que DROP_COL foi removida
        assert 'DROP_COL' not in result.columns
        assert 'Close' in result.columns
        assert 'Volume' in result.columns

    @patch('utils.trainer.SequenceDataset')
    def test_create_datasets(self, mock_sequence_dataset_class, trainer_instance, sample_dataframe, hyperparams):
        """Testa criação de datasets de treino, validação e teste"""
        # Mock do dataset
        mock_dataset = MagicMock()
        mock_feature_scaler = MagicMock()
        mock_target_scaler = MagicMock()
        mock_dataset.get_scalers.return_value = (mock_feature_scaler, mock_target_scaler)

        # Configurar X e y como arrays numpy para simular split
        mock_dataset.X = np.random.randn(140, 30, 4)  # 140 samples de treino
        mock_dataset.y = np.random.randn(140)

        mock_sequence_dataset_class.return_value = mock_dataset

        result = trainer_instance._create_datasets(sample_dataframe, hyperparams)

        train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = result

        # Verificar que SequenceDataset foi chamado 3 vezes (train, val, test)
        assert mock_sequence_dataset_class.call_count == 3

        # Verificar que scalers foram retornados
        assert feature_scaler == mock_feature_scaler
        assert target_scaler == mock_target_scaler

    @patch('utils.trainer.EarlyStopping')
    @patch('utils.trainer.ModelCheckpoint')
    @patch('utils.trainer.LearningRateMonitor')
    def test_create_callbacks(self, mock_lr_monitor, mock_checkpoint, mock_early_stopping,
                             trainer_instance, hyperparams):
        """Testa criação de callbacks do PyTorch Lightning"""
        # Configurar mocks
        mock_early_stopping_instance = MagicMock()
        mock_checkpoint_instance = MagicMock()
        mock_lr_monitor_instance = MagicMock()

        mock_early_stopping.return_value = mock_early_stopping_instance
        mock_checkpoint.return_value = mock_checkpoint_instance
        mock_lr_monitor.return_value = mock_lr_monitor_instance

        # Simular temp_dir
        trainer_instance.temp_dir = "/tmp/test_dir"

        result = trainer_instance._create_callbacks(hyperparams)

        assert len(result) == 3
        assert result[0] == mock_early_stopping_instance
        assert result[1] == mock_checkpoint_instance
        assert result[2] == mock_lr_monitor_instance

        # Verificar parâmetros do EarlyStopping
        mock_early_stopping.assert_called_once_with(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            mode='min',
            verbose=False
        )

        # Verificar parâmetros do ModelCheckpoint
        mock_checkpoint.assert_called_once_with(
            dirpath="/tmp/test_dir",
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=False
        )

        # Verificar parâmetros do LearningRateMonitor
        mock_lr_monitor.assert_called_once_with(logging_interval='epoch')

    @patch('utils.trainer.logger')
    def test_calculate_all_metrics(self, mock_logger, trainer_instance, mock_model):
        """Testa cálculo de métricas de validação e teste"""
        # Configurar mocks
        device = MagicMock()
        mock_param = MagicMock()
        mock_param.device = device
        mock_model.parameters.return_value = iter([mock_param])  # Retornar iterador

        # Mock dataloaders
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()

        # Mock batches - usar tensores mock
        mock_val_inputs = MagicMock()
        mock_val_targets = MagicMock()
        mock_test_inputs = MagicMock()
        mock_test_targets = MagicMock()
        
        # Configurar método .to() para retornar o próprio objeto
        mock_val_inputs.to.return_value = mock_val_inputs
        mock_val_targets.to.return_value = mock_val_targets
        mock_test_inputs.to.return_value = mock_test_inputs
        mock_test_targets.to.return_value = mock_test_targets
        
        val_batch = (mock_val_inputs, mock_val_targets)
        test_batch = (mock_test_inputs, mock_test_targets)

        mock_val_loader.__iter__.return_value = [val_batch]
        mock_test_loader.__iter__.return_value = [test_batch]

        # Mock scalers
        mock_target_scaler = MagicMock()
        mock_target_scaler.inverse_transform.return_value = np.array([[25.0], [26.0]] * 16)

        # Mock datasets
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()

        with patch('utils.trainer.calculate_metrics') as mock_calc_metrics:
            # Configurar o mock para aceitar qualquer argumento
            mock_calc_metrics.return_value = {"mae": 1.0, "rmse": 1.5}
            
            result = trainer_instance._calculate_all_metrics(
                mock_model, mock_val_loader, mock_test_loader,
                mock_val_dataset, mock_test_dataset, mock_target_scaler
            )

            expected = {
                "validation": {"mae": 1.0, "rmse": 1.5},
                "test": {"mae": 1.0, "rmse": 1.5}
            }
            assert result == expected

            # Verificar que calculate_metrics foi chamado 2 vezes
            assert mock_calc_metrics.call_count == 2

    @patch('builtins.open', new_callable=MagicMock)
    @patch('utils.trainer.os.path.exists')
    def test_serialize_model_with_checkpoint(self, mock_exists, mock_open, trainer_instance):
        """Testa serialização de modelo com checkpoint existente"""
        mock_exists.return_value = True
        checkpoint_path = "/tmp/checkpoint.ckpt"
        expected_bytes = b"model_data"

        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = expected_bytes

        result = trainer_instance._serialize_model(MagicMock(), checkpoint_path)

        assert result == expected_bytes
        mock_open.assert_called_once_with(checkpoint_path, 'rb')

    @patch('utils.trainer.io.BytesIO')
    @patch('utils.trainer.torch.save')
    def test_serialize_model_without_checkpoint(self, mock_torch_save, mock_bytesio, trainer_instance):
        """Testa serialização de modelo sem checkpoint"""
        mock_model = MagicMock()
        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.read.return_value = b"model_bytes"

        result = trainer_instance._serialize_model(mock_model)

        assert result == b"model_bytes"
        mock_torch_save.assert_called_once()

    @patch('utils.trainer.joblib.dump')
    @patch('utils.trainer.io.BytesIO')
    def test_serialize_scalers(self, mock_bytesio, mock_joblib_dump, trainer_instance):
        """Testa serialização de scalers"""
        mock_feature_scaler = MagicMock()
        mock_target_scaler = MagicMock()

        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.read.return_value = b"scaler_bytes"

        result = trainer_instance._serialize_scalers(mock_feature_scaler, mock_target_scaler)

        assert result == b"scaler_bytes"

        # Verificar que joblib.dump foi chamado com os scalers corretos
        call_args = mock_joblib_dump.call_args
        scalers_dict = call_args[0][0]  # Primeiro argumento posicional
        assert 'feature_scaler' in scalers_dict
        assert 'target_scaler' in scalers_dict

    @patch('utils.trainer.joblib.dump')
    @patch('utils.trainer.io.BytesIO')
    def test_serialize_metrics(self, mock_bytesio, mock_joblib_dump, trainer_instance, hyperparams):
        """Testa serialização de métricas"""
        metrics = {
            "validation": {"mae": 1.0},
            "test": {"mae": 1.2}
        }

        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.read.return_value = b"metrics_bytes"

        result = trainer_instance._serialize_metrics(metrics, hyperparams)

        assert result == b"metrics_bytes"

        # Verificar que joblib.dump foi chamado com estrutura correta
        call_args = mock_joblib_dump.call_args
        metrics_dict = call_args[0][0]
        assert 'validacao' in metrics_dict
        assert 'teste' in metrics_dict
        assert 'config' in metrics_dict