import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_real, predictions, dataset_name=""):
    """
    Calcula métricas de avaliação do modelo
    
    Args:
        y_real: Valores reais (array-like)
        predictions: Valores preditos (array-like)
        dataset_name: Nome do dataset (para logging)
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    try:
        # Converter para numpy arrays se necessário
        y_real = np.array(y_real)
        predictions = np.array(predictions)
        
        # MAE - Mean Absolute Error
        mae = mean_absolute_error(y_real, predictions)
        
        # MSE - Mean Squared Error
        mse = mean_squared_error(y_real, predictions)
        
        # RMSE - Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # R² - Coeficiente de Determinação
        r2 = r2_score(y_real, predictions)
        
        # MAPE - Mean Absolute Percentage Error
        # Evita divisão por zero usando np.where
        mape = np.mean(np.abs((y_real - predictions) / np.where(y_real != 0, y_real, 1))) * 100
        
        # Acurácia Direcional
        mean_y = np.mean(y_real)
        real_direction = np.sign(y_real - mean_y)
        pred_direction = np.sign(predictions - mean_y)
        directional_accuracy = (real_direction == pred_direction).mean()
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        }
        
        if dataset_name:
            logger.info(f"\n📊 Métricas {dataset_name}:")
            logger.info(f"   MAE:  R$ {mae:.3f}")
            logger.info(f"   RMSE: R$ {rmse:.3f}")
            logger.info(f"   MAPE: {mape:.2f}%")
            logger.info(f"   R²:   {r2:.4f} ({r2*100:.2f}%)")
            logger.info(f"   Acurácia Direcional: {directional_accuracy*100:.2f}%")
        
        return metrics
    
    except Exception as e:
        logger.exception(f"Erro ao calcular métricas para {dataset_name}")
        raise
