import sys
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Load the yfinance_client module dynamically
yfc_path = Path(__file__).resolve().parents[3] / "data-service" / "utils" / "yfinance_client.py"
spec = importlib.util.spec_from_file_location("yfinance_client", str(yfc_path))
yfc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yfc_module)
YFinanceClient = yfc_module.YFinanceClient


@pytest.fixture
def yfinance_client():
    """Create YFinanceClient instance"""
    return YFinanceClient(max_retries=3, backoff=0.1)


class TestYFinanceClient:

    def test_init(self):
        """Test YFinanceClient initialization"""
        client = YFinanceClient(max_retries=5, backoff=2.0)
        assert client.max_retries == 5
        assert client.backoff == 2.0
        assert hasattr(client, 'logger')
        assert client.logger.name == 'yfinance_client'

    def test_init_defaults(self):
        """Test YFinanceClient initialization with defaults"""
        client = YFinanceClient()
        assert client.max_retries == 3
        assert client.backoff == 1.0

    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_success(self, mock_yf, yfinance_client):
        """Test successful data fetch"""
        # Mock the ticker and history
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))

        mock_ticker.history.return_value = sample_data

        with patch.object(yfinance_client.logger, 'info') as mock_info:
            result = yfinance_client.fetch_ticker_data('PETR4', '2023-01-01', '2023-01-04')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_yf.Ticker.assert_called_once_with('PETR4.SA')
        mock_ticker.history.assert_called_once_with(
            start='2023-01-01',
            end='2023-01-04',
            interval='1d',
            auto_adjust=False
        )
        mock_info.assert_called_once_with('PETR4: 3 registros encontrados')

    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_empty_data(self, mock_yf, yfinance_client):
        """Test when yfinance returns empty data"""
        # Mock the ticker and history
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        mock_ticker.history.return_value = pd.DataFrame()

        with patch.object(yfinance_client.logger, 'warning') as mock_warning:
            result = yfinance_client.fetch_ticker_data('INVALID', '2023-01-01', '2023-01-04')

        assert result is None
        mock_warning.assert_called_once_with('Sem dados para INVALID')

    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_none_data(self, mock_yf, yfinance_client):
        """Test when yfinance returns None"""
        # Mock the ticker and history
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        mock_ticker.history.return_value = None

        with patch.object(yfinance_client.logger, 'warning') as mock_warning:
            result = yfinance_client.fetch_ticker_data('INVALID', '2023-01-01', '2023-01-04')

        assert result is None
        mock_warning.assert_called_once_with('Sem dados para INVALID')

    @patch.object(yfc_module, 'time')
    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_retry_success(self, mock_yf, mock_time, yfinance_client):
        """Test successful fetch after retries"""
        # Mock the ticker
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        # First two calls fail, third succeeds
        sample_data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2))

        mock_ticker.history.side_effect = [Exception("Network error"), Exception("Timeout"), sample_data]

        with patch.object(yfinance_client.logger, 'warning') as mock_warning, \
             patch.object(yfinance_client.logger, 'info') as mock_info:

            result = yfinance_client.fetch_ticker_data('PETR4', '2023-01-01', '2023-01-03')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert mock_ticker.history.call_count == 3
        assert mock_time.sleep.call_count == 2
        # Check sleep times: backoff * (2 ** attempt)
        mock_time.sleep.assert_any_call(0.1 * (2 ** 0))  # 0.1
        mock_time.sleep.assert_any_call(0.1 * (2 ** 1))  # 0.2
        mock_warning.assert_called()
        mock_info.assert_called_once_with('PETR4: 2 registros encontrados')

    @patch.object(yfc_module, 'yf')
    @patch.object(yfc_module, 'time')
    def test_fetch_ticker_data_max_retries_exceeded(self, mock_time, mock_yf, yfinance_client):
        """Test when all retries are exhausted"""
        # Mock the ticker
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        # All calls fail
        mock_ticker.history.side_effect = Exception("Persistent error")

        with patch.object(yfinance_client.logger, 'warning') as mock_warning, \
             patch.object(yfinance_client.logger, 'error') as mock_error:

            with pytest.raises(Exception) as exc_info:
                yfinance_client.fetch_ticker_data('PETR4', '2023-01-01', '2023-01-03')

        assert str(exc_info.value) == "Persistent error"
        assert mock_ticker.history.call_count == 3
        assert mock_time.sleep.call_count == 2
        mock_warning.assert_called()
        mock_error.assert_called_once_with('Falha ao buscar PETR4 após 3 tentativas')

    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_custom_interval(self, mock_yf, yfinance_client):
        """Test fetch with custom interval"""
        # Mock the ticker and history
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        sample_data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2))

        mock_ticker.history.return_value = sample_data

        with patch.object(yfinance_client.logger, 'info'):
            result = yfinance_client.fetch_ticker_data('PETR4', '2023-01-01', '2023-01-03', interval='1h')

        mock_ticker.history.assert_called_once_with(
            start='2023-01-01',
            end='2023-01-03',
            interval='1h',
            auto_adjust=False
        )

    @patch.object(yfc_module, 'yf')
    def test_fetch_ticker_data_empty_ticker(self, mock_yf, yfinance_client):
        """Test when ticker is empty string"""
        mock_ticker = Mock()
        mock_yf.Ticker.return_value = mock_ticker

        mock_ticker.history.return_value = pd.DataFrame()

        with patch.object(yfinance_client.logger, 'warning') as mock_warning:
            result = yfinance_client.fetch_ticker_data('', '2023-01-01', '2023-01-04')

        assert result is None
        mock_yf.Ticker.assert_called_once_with('.SA')
        mock_warning.assert_called_once_with('Sem dados para ')