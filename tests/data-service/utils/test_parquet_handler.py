import sys
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Load the parquet_handler module dynamically
ph_path = Path(__file__).resolve().parents[3] / "data-service" / "utils" / "parquet_handler.py"
spec = importlib.util.spec_from_file_location("parquet_handler", str(ph_path))
ph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph_module)
ParquetHandler = ph_module.ParquetHandler


@pytest.fixture
def mock_container_client():
    """Create a mock container client"""
    return Mock()


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing"""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = {
        'Close': [100, 101, 102, 103, 104],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def parquet_handler(mock_container_client):
    """Create ParquetHandler instance with mock client"""
    return ParquetHandler(mock_container_client)


class TestParquetHandler:

    def test_init(self, mock_container_client):
        """Test ParquetHandler initialization"""
        handler = ParquetHandler(mock_container_client)
        assert handler.container_client == mock_container_client
        assert hasattr(handler, 'logger')
        assert handler.logger.name == 'parquet_handler'
        assert handler.tz.key == 'America/Sao_Paulo'

    def test_save_daily_data(self, parquet_handler, sample_df):
        """Test saving daily data with partitioning"""
        test_date = datetime(2023, 1, 1)
        ticker = 'TEST'

        with patch.object(parquet_handler, '_upload_parquet') as mock_upload:
            result = parquet_handler.save_daily_data(sample_df, ticker, test_date)

            # Check return path
            assert result == '2023/01/01/TEST.parquet'

            # Check that _upload_parquet was called
            mock_upload.assert_called_once()
            args, kwargs = mock_upload.call_args
            df_arg = args[0]
            path_arg = args[1]

            # Check DataFrame modifications
            assert 'execution_timestamp' in df_arg.columns
            assert 'ticker' in df_arg.columns
            assert (df_arg['ticker'] == ticker).all()
            assert path_arg == '2023/01/01/TEST.parquet'

    def test_save_history_data(self, parquet_handler, sample_df):
        """Test saving history data"""
        ticker = 'TEST'

        with patch.object(parquet_handler, '_upload_parquet') as mock_upload:
            result = parquet_handler.save_history_data(sample_df, ticker)

            # Check return path
            assert result == 'history/TEST.parquet'

            # Check that _upload_parquet was called
            mock_upload.assert_called_once()
            args, kwargs = mock_upload.call_args
            df_arg = args[0]
            path_arg = args[1]

            # Check DataFrame modifications
            assert 'execution_timestamp' in df_arg.columns
            assert 'ticker' in df_arg.columns
            assert (df_arg['ticker'] == ticker).all()
            assert 'index' in df_arg.columns  # reset_index adds index column
            assert path_arg == 'history/TEST.parquet'

    def test_upload_parquet(self, parquet_handler, sample_df):
        """Test _upload_parquet method"""
        blob_path = 'test/path/file.parquet'

        # Mock the blob client
        mock_blob_client = Mock()
        parquet_handler.container_client.get_blob_client.return_value = mock_blob_client

        parquet_handler._upload_parquet(sample_df, blob_path)

        # Check that get_blob_client was called with correct path
        parquet_handler.container_client.get_blob_client.assert_called_once_with(blob_path)

        # Check that upload_blob was called
        mock_blob_client.upload_blob.assert_called_once()
        args, kwargs = mock_blob_client.upload_blob.call_args

        # Check that it's a BytesIO object
        assert hasattr(args[0], 'read')
        assert kwargs['overwrite'] is True

    def test_save_daily_data_modifies_copy(self, parquet_handler, sample_df):
        """Test that save_daily_data doesn't modify the original DataFrame"""
        original_columns = sample_df.columns.tolist()
        original_len = len(sample_df)

        test_date = datetime(2023, 1, 1)
        ticker = 'TEST'

        with patch.object(parquet_handler, '_upload_parquet'):
            parquet_handler.save_daily_data(sample_df, ticker, test_date)

        # Original DataFrame should be unchanged
        assert sample_df.columns.tolist() == original_columns
        assert len(sample_df) == original_len
        assert 'execution_timestamp' not in sample_df.columns
        assert 'ticker' not in sample_df.columns

    def test_save_history_data_modifies_copy(self, parquet_handler, sample_df):
        """Test that save_history_data doesn't modify the original DataFrame"""
        original_columns = sample_df.columns.tolist()
        original_index_name = sample_df.index.name

        ticker = 'TEST'

        with patch.object(parquet_handler, '_upload_parquet'):
            parquet_handler.save_history_data(sample_df, ticker)

        # Original DataFrame should be unchanged
        assert sample_df.columns.tolist() == original_columns
        assert sample_df.index.name == original_index_name
        assert 'execution_timestamp' not in sample_df.columns
        assert 'ticker' not in sample_df.columns
        assert 'index' not in sample_df.columns