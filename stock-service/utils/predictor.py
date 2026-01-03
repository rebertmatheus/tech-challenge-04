import torch
import numpy as np
import pandas as pd
import logging
import io
import tempfile
import os

from .stocks_lstm import StocksLSTM

logger = logging.getLogger(__name__)

def prepare_prediction_sequence(df: pd.DataFrame, hyperparams: dict, feature_scaler):
    """
    Prepara sequência temporal para predição
    
    Args:
        df: DataFrame com dados (já com DROP_COLUMNS aplicado)
        hyperparams: Dicionário com hiperparâmetros (SEQUENCE_LENGTH, FEATURE_COLS)
        feature_scaler: Scaler pré-treinado para features
    
    Returns:
        torch.Tensor: Tensor com shape (1, SEQUENCE_LENGTH, num_features)
    """
    try:
        sequence_length = hyperparams["SEQUENCE_LENGTH"]
        feature_cols = hyperparams["FEATURE_COLS"]
        
        # Validar que temos dados suficientes
        if len(df) < sequence_length:
            raise ValueError(
                f"Dados insuficientes: {len(df)} registros, "
                f"necessário pelo menos {sequence_length} para criar sequência"
            )
        
        # Normalizar features
        scaled_features = feature_scaler.transform(df[feature_cols])
        
        # Pegar última sequência (últimos SEQUENCE_LENGTH registros)
        last_sequence = scaled_features[-sequence_length:]
        
        # Converter para tensor: (1, SEQUENCE_LENGTH, num_features)
        sequence_tensor = torch.tensor(
            last_sequence.reshape(1, sequence_length, len(feature_cols)),
            dtype=torch.float32
        )
        
        logger.info(
            f"Sequência preparada: shape {sequence_tensor.shape} "
            f"(1, {sequence_length}, {len(feature_cols)})"
        )
        
        return sequence_tensor
    
    except Exception as e:
        logger.exception("Erro ao preparar sequência para predição")
        raise

def load_model_from_bytes(model_bytes: bytes, hyperparams: dict):
    """
    Carrega modelo PyTorch Lightning a partir de bytes
    
    Args:
        model_bytes: Bytes do checkpoint do modelo
        hyperparams: Dicionário com hiperparâmetros
    
    Returns:
        StocksLSTM: Modelo carregado
    """
    try:
        # Criar arquivo temporário para carregar o checkpoint
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "model.ckpt")
        
        try:
            # Escrever bytes em arquivo temporário
            with open(temp_path, 'wb') as f:
                f.write(model_bytes)
            
            # Carregar modelo do checkpoint
            model = StocksLSTM.load_from_checkpoint(temp_path, config=hyperparams)
            
            logger.info("Modelo carregado do checkpoint")
            return model
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    except Exception as e:
        logger.exception("Erro ao carregar modelo de bytes")
        raise

def predict_price(model: StocksLSTM, sequence: torch.Tensor, target_scaler):
    """
    Executa predição de preço usando o modelo
    
    Args:
        model: Modelo StocksLSTM carregado
        sequence: Tensor com sequência temporal (1, SEQUENCE_LENGTH, num_features)
        target_scaler: Scaler pré-treinado para target
    
    Returns:
        float: Preço predito (desnormalizado)
    """
    try:
        # Mover modelo e sequência para o mesmo device
        device = next(model.parameters()).device
        sequence = sequence.to(device)
        
        # Colocar modelo em modo avaliação
        model.eval()
        
        # Executar predição (sem gradientes)
        with torch.no_grad():
            prediction_normalized = model(sequence).flatten()
        
        # Desnormalizar predição
        prediction_array = prediction_normalized.cpu().numpy().reshape(-1, 1)
        prediction_real = target_scaler.inverse_transform(prediction_array)
        
        predicted_price = float(prediction_real[0][0])
        
        logger.info(f"Predição realizada: R$ {predicted_price:.2f}")
        
        return predicted_price
    
    except Exception as e:
        logger.exception("Erro ao executar predição")
        raise

