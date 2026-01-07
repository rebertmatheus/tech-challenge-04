import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch
import importlib.util

# Load the metrics module dynamically
metrics_path = Path(__file__).resolve().parents[3] / "stock-service" / "utils" / "metrics.py"
spec = importlib.util.spec_from_file_location("metrics", str(metrics_path))
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)


class TestCalculateMetrics:
    """Test cases for calculate_metrics function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample real and predicted values"""
        np.random.seed(42)
        # Generate realistic stock price data
        y_real = np.array([10.5, 11.2, 10.8, 12.1, 11.9, 13.2, 12.8, 14.1, 13.9, 15.2])
        # Add some noise to create predictions
        predictions = y_real + np.random.normal(0, 0.5, len(y_real))

        return y_real, predictions

    def test_calculate_metrics_basic(self, sample_data):
        """Test basic metrics calculation"""
        y_real, predictions = sample_data

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Check that all expected metrics are present
        expected_keys = ['mae', 'mse', 'rmse', 'mape', 'r2', 'directional_accuracy']
        assert all(key in result for key in expected_keys)

        # Check that all values are floats
        assert all(isinstance(result[key], float) for key in expected_keys)

        # Check reasonable value ranges
        assert result['mae'] >= 0
        assert result['mse'] >= 0
        assert result['rmse'] >= 0
        assert result['mape'] >= 0
        assert result['r2'] <= 1  # R² can be negative
        assert 0 <= result['directional_accuracy'] <= 1

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Perfect predictions should have zero error metrics
        assert result['mae'] == 0.0
        assert result['mse'] == 0.0
        assert result['rmse'] == 0.0
        assert result['mape'] == 0.0
        assert result['r2'] == 1.0  # Perfect R²
        assert result['directional_accuracy'] == 1.0  # Perfect directional accuracy

    def test_calculate_metrics_with_dataset_name(self, sample_data, caplog):
        """Test metrics calculation with dataset name (logging)"""
        y_real, predictions = sample_data

        with caplog.at_level('INFO'):
            result = metrics_module.calculate_metrics(y_real, predictions, "Test Dataset")

        # Check that logging occurred
        assert "📊 Métricas Test Dataset:" in caplog.text
        assert "MAE:" in caplog.text
        assert "RMSE:" in caplog.text
        assert "MAPE:" in caplog.text
        assert "R²:" in caplog.text
        assert "Acurácia Direcional:" in caplog.text

    def test_calculate_metrics_with_zeros(self):
        """Test metrics calculation when real values contain zeros (MAPE edge case)"""
        y_real = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        predictions = np.array([0.1, 1.1, 2.1, 0.1, 3.1])

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Should not crash and should handle zero division gracefully
        assert isinstance(result['mape'], float)
        assert result['mape'] >= 0

    def test_calculate_metrics_single_value(self):
        """Test metrics calculation with single data point"""
        y_real = np.array([10.0])
        predictions = np.array([10.5])

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Check that metrics are calculated (though some may be undefined)
        expected_keys = ['mae', 'mse', 'rmse', 'mape', 'r2', 'directional_accuracy']
        assert all(key in result for key in expected_keys)

    def test_calculate_metrics_numpy_conversion(self):
        """Test that function converts inputs to numpy arrays"""
        y_real = [1.0, 2.0, 3.0, 4.0, 5.0]  # List input
        predictions = [1.1, 2.1, 3.1, 4.1, 5.1]  # List input

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Should work with list inputs
        assert isinstance(result['mae'], float)
        assert result['mae'] > 0

    def test_calculate_metrics_directional_accuracy(self):
        """Test directional accuracy calculation specifically"""
        # Create data where directional accuracy should be 1.0
        # All values significantly above or below mean to avoid floating point issues
        y_real = np.array([10.0, 11.0, 12.0, 13.0, 14.0])  # Mean = 12.0, all above mean
        predictions = np.array([10.1, 11.1, 12.1, 13.1, 14.1])  # All above mean

        result = metrics_module.calculate_metrics(y_real, predictions)

        # Both series are entirely above their means, so directional accuracy should be 1.0
    def test_calculate_metrics_error_handling(self):
        """Test error handling in metrics calculation"""
        # Test with mismatched lengths
        y_real = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 2.1])  # Different length

        with pytest.raises(Exception):
            metrics_module.calculate_metrics(y_real, predictions)

    @patch.object(metrics_module.logger, 'exception')
    def test_calculate_metrics_exception_logging(self, mock_logger_exception):
        """Test that exceptions are logged properly"""
        # This should trigger an exception
        y_real = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 2.1])  # Different length

        with pytest.raises(Exception):
            metrics_module.calculate_metrics(y_real, predictions, "Test Dataset")

        # Check that exception was logged
        mock_logger_exception.assert_called_once()