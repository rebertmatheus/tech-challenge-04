import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import importlib.util

# Load the feature_engineering module dynamically
fe_path = Path(__file__).resolve().parents[3] / "data-service" / "utils" / "feature_engineering.py"
spec = importlib.util.spec_from_file_location("feature_engineering", str(fe_path))
fe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_module)
FeatureEngineer = fe_module.FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    np.random.seed(42)
    
    data = {
        'Open': np.random.uniform(100, 110, 30),
        'High': np.random.uniform(105, 115, 30),
        'Low': np.random.uniform(95, 105, 30),
        'Close': np.random.uniform(100, 110, 30),
        'Adj Close': np.random.uniform(100, 110, 30),
        'Volume': np.random.randint(100000, 1000000, 30)
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def minimal_data():
    """Create minimal data with only required columns"""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    
    data = {
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'Adj Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'Volume': [100000] * 10
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


class TestFeatureEngineer:
    
    def test_init(self):
        """Test FeatureEngineer initialization"""
        fe = FeatureEngineer()
        assert hasattr(fe, 'logger')
        assert fe.logger.name == 'feature_engineering'
    
    def test_create_features_none_input(self):
        """Test create_features with None input"""
        fe = FeatureEngineer()
        with patch.object(fe.logger, 'warning') as mock_warning:
            result = fe.create_features(None)
            assert result is None
            mock_warning.assert_called_once()
    
    def test_create_features_invalid_type(self):
        """Test create_features with invalid input type"""
        fe = FeatureEngineer()
        with pytest.raises(TypeError):
            fe.create_features("not a dataframe")
    
    def test_create_features_missing_required_columns(self):
        """Test create_features with missing required columns"""
        fe = FeatureEngineer()
        df = pd.DataFrame({'Close': [1, 2, 3]})
        
        with pytest.raises(ValueError) as exc_info:
            fe.create_features(df)
        
        assert "Colunas obrigatórias não encontradas" in str(exc_info.value)
    
    def test_create_features_minimal_data_training(self, minimal_data):
        """Test create_features with minimal data in training mode"""
        fe = FeatureEngineer()
        
        with patch.object(fe.logger, 'info') as mock_info:
            result = fe.create_features(minimal_data, is_training_data=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(minimal_data)  # Some rows removed due to NaN
        
        # Check that basic features are created
        expected_features = ['rsi_7', 'rsi_14', 'macd_histogram', 'ma3', 'ma5', 'ma9', 
                           'distance_ma3', 'distance_ma9', 'return_1d', 'return_3d', 
                           'roc_3', 'volatility_5d', 'volatility_ratio', 'relative_volume', 
                           'volume_ratio_5', 'bb_position', 'target']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check target is present for training
        assert 'target' in result.columns
    
    def test_create_features_minimal_data_prediction(self, minimal_data):
        """Test create_features with minimal data in prediction mode"""
        fe = FeatureEngineer()
        
        with patch.object(fe.logger, 'info') as mock_info:
            result = fe.create_features(minimal_data, is_training_data=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(minimal_data)  # No rows removed in prediction mode
        
        # Check that features are created but no target
        expected_features = ['rsi_7', 'rsi_14', 'macd_histogram', 'ma3', 'ma5', 'ma9', 
                           'distance_ma3', 'distance_ma9', 'return_1d', 'return_3d', 
                           'roc_3', 'volatility_5d', 'volatility_ratio', 'relative_volume', 
                           'volume_ratio_5', 'bb_position']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check no target in prediction mode
        assert 'target' not in result.columns
    
    def test_create_features_full_ohlc_data(self, sample_data):
        """Test create_features with full OHLC data"""
        fe = FeatureEngineer()
        
        with patch.object(fe.logger, 'info') as mock_info:
            result = fe.create_features(sample_data, is_training_data=True)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check OHLC-specific features
        assert 'stoch_k' in result.columns
        assert 'gap' in result.columns
        
        # Verify stochastic is calculated (not NaN)
        assert not result['stoch_k'].isna().all()
        assert not result['gap'].isna().all()
    
    def test_create_features_no_ohlc(self, minimal_data):
        """Test create_features without OHLC data"""
        fe = FeatureEngineer()
        
        with patch.object(fe.logger, 'warning') as mock_warning:
            result = fe.create_features(minimal_data, is_training_data=True)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check OHLC features are not present or are zero
        assert 'stoch_k' not in result.columns
        assert 'gap' in result.columns
        assert (result['gap'] == 0.0).all()
    
    def test_create_features_multiindex(self, sample_data):
        """Test create_features with MultiIndex columns"""
        fe = FeatureEngineer()
        
        # Create MultiIndex with correct structure (level 0: metric, level 1: ticker)
        multi_df = pd.DataFrame()
        multi_df[('Open', 'PETR4')] = sample_data['Open']
        multi_df[('High', 'PETR4')] = sample_data['High']
        multi_df[('Low', 'PETR4')] = sample_data['Low']
        multi_df[('Close', 'PETR4')] = sample_data['Close']
        multi_df[('Adj Close', 'PETR4')] = sample_data['Adj Close']
        multi_df[('Volume', 'PETR4')] = sample_data['Volume']
        multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)
        
        with patch.object(fe.logger, 'info') as mock_info:
            result = fe.create_features(multi_df, is_training_data=True)
        
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result.columns, pd.MultiIndex)
        
        # Check features are created
        assert 'rsi_7' in result.columns
    
    def test_create_features_exception_handling(self, minimal_data):
        """Test exception handling in create_features"""
        fe = FeatureEngineer()
        
        # Force an exception by making data problematic
        bad_data = minimal_data.copy()
        bad_data['Adj Close'] = 'not_numeric'  # This should cause issues
        
        with patch.object(fe.logger, 'exception') as mock_exception:
            with pytest.raises(Exception):
                fe.create_features(bad_data)
            
            mock_exception.assert_called_once()

    def test_create_features_series_input_conversion(self):
        """Testa conversão de Series para DataFrame no input"""
        fe = FeatureEngineer()
        
        # Criar uma Series (que será convertida para DataFrame)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], 
                          index=dates, name='Close')
        
        # Como Series não é DataFrame, deve falhar com TypeError
        with pytest.raises(TypeError, match="Esperado pd.DataFrame"):
            fe.create_features(series)