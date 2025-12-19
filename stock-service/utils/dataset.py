import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """
    Dataset com FEATURES E TARGET normalizados
    Cria sequências temporais para treinamento LSTM
    """
    
    def __init__(self, df, sequence_length, feature_cols, target_col,
                 feature_scaler=None, target_scaler=None, fit_scalers=True):
        """
        Args:
            df: DataFrame pandas com dados históricos
            sequence_length: Tamanho da sequência temporal
            feature_cols: Lista de colunas de features
            target_col: Nome da coluna target
            feature_scaler: Scaler pré-treinado para features (se fit_scalers=False)
            target_scaler: Scaler pré-treinado para target (se fit_scalers=False)
            fit_scalers: Se True, cria novos scalers. Se False, usa os fornecidos
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Normalizar FEATURES
        if fit_scalers:
            self.feature_scaler = MinMaxScaler()
            scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
        else:
            self.feature_scaler = feature_scaler
            scaled_features = self.feature_scaler.transform(df[feature_cols])
        
        # Normalizar TARGET
        if fit_scalers:
            self.target_scaler = MinMaxScaler()
            scaled_targets = self.target_scaler.fit_transform(df[[target_col]])
        else:
            self.target_scaler = target_scaler
            scaled_targets = self.target_scaler.transform(df[[target_col]])
        
        self.features = scaled_features
        self.targets = scaled_targets.squeeze()
        
        # Criar sequências
        self.X, self.y = self._create_sequences()
        
        logger.info(f"Dataset criado: {len(self.X)} sequências de tamanho {sequence_length}")
    
    def _create_sequences(self):
        """Cria sequências temporais a partir dos dados normalizados"""
        X, y = [], []
        for i in range(self.sequence_length, len(self.features)):
            X.append(self.features[i - self.sequence_length:i])
            y.append(self.targets[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    def get_scalers(self):
        """Retorna os scalers de features e target"""
        return self.feature_scaler, self.target_scaler
