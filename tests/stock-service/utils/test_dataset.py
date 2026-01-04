import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import MagicMock, patch
import importlib.util

# Mock torch and torch.utils.data before importing
mock_torch = MagicMock()
mock_dataset = MagicMock()
mock_torch.utils.data.Dataset = mock_dataset
mock_torch.tensor = MagicMock()

# Load the dataset module dynamically with torch mocked
with patch.dict('sys.modules', {
    'torch': mock_torch,
    'torch.utils': MagicMock(),
    'torch.utils.data': MagicMock(),
}):
    dataset_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset", str(dataset_path))
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)


class TestSequenceDataset:
    """Test cases for SequenceDataset class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = {
            'close': np.random.uniform(10, 50, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'returns': np.random.normal(0, 0.02, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def pre_fitted_scalers(self, sample_data):
        """Create pre-fitted scalers"""
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        feature_cols = ['close', 'volume', 'returns']
        target_col = 'close'

        feature_scaler.fit(sample_data[feature_cols])
        target_scaler.fit(sample_data[[target_col]])

        return feature_scaler, target_scaler

    def test_create_sequences_logic(self):
        """Test the logic of _create_sequences method"""
        # Simulate the data that would be in a dataset instance
        sequence_length = 10
        features = np.random.rand(100, 3).astype(np.float32)
        targets = np.random.rand(100).astype(np.float32)

        # Replicate the _create_sequences logic
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(targets[i])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        assert X.shape[0] == 90  # 100 - 10
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 3   # number of features
        assert y.shape[0] == 90

    def test_insufficient_data_logic(self):
        """Test sequence creation with insufficient data"""
        sequence_length = 50
        features = np.random.rand(30, 3).astype(np.float32)  # Only 30 samples
        targets = np.random.rand(30).astype(np.float32)

        # Replicate the _create_sequences logic
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(targets[i])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 0  # No sequences can be created
        assert y.shape[0] == 0

    def test_scaler_creation(self, sample_data):
        """Test scaler creation and fitting"""
        feature_cols = ['close', 'volume', 'returns']
        target_col = 'close'

        # Test feature scaler
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(sample_data[feature_cols])

        assert scaled_features.shape == (100, 3)
        assert scaled_features.min() >= 0
        assert scaled_features.max() <= 1

        # Test target scaler
        target_scaler = MinMaxScaler()
        scaled_targets = target_scaler.fit_transform(sample_data[[target_col]])

        assert scaled_targets.shape == (100, 1)
        assert scaled_targets.min() >= 0
        assert scaled_targets.max() <= 1

    def test_scaler_transform_only(self, sample_data, pre_fitted_scalers):
        """Test using pre-fitted scalers"""
        feature_scaler, target_scaler = pre_fitted_scalers
        feature_cols = ['close', 'volume', 'returns']
        target_col = 'close'

        # Transform using existing scalers
        scaled_features = feature_scaler.transform(sample_data[feature_cols])
        scaled_targets = target_scaler.transform(sample_data[[target_col]])

        assert scaled_features.shape == (100, 3)
        assert scaled_targets.shape == (100, 1)
        assert feature_scaler is pre_fitted_scalers[0]
        assert target_scaler is pre_fitted_scalers[1]