from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import logging
import json
import pandas as pd
import io
import joblib

logger = logging.getLogger(__name__)

def get_storage_client(conn_str: str, container_name: str):
    """Retorna container client configurado"""
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)

        try:
            container_client.create_container()
            logger.info(f"Container {container_name} criado")
        except ResourceExistsError:
            logger.info(f"Container {container_name} já existe")

        return container_client
    except Exception as e:
        logger.exception("Falha ao conectar no Azure Blob Storage")
        raise

def load_hyperparameters(container_client, ticker: str, version: str = None):
    """
    Carrega hiperparâmetros do Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão dos hiperparâmetros (opcional, ex: "v1")
    
    Returns:
        dict: Dicionário com hiperparâmetros
    """
    try:
        if version:
            blob_path = f"hyperparameters/{ticker}_v{version}.json"
        else:
            blob_path = f"hyperparameters/{ticker}.json"
        
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            if version:
                # Tenta sem versão
                blob_path = f"hyperparameters/{ticker}.json"
                blob_client = container_client.get_blob_client(blob_path)
                if not blob_client.exists():
                    raise FileNotFoundError(f"Hiperparâmetros não encontrados para {ticker}_v{version} ou {ticker}")
            else:
                raise FileNotFoundError(f"Hiperparâmetros não encontrados para {ticker}")
        
        blob_data = blob_client.download_blob().readall()
        hyperparams = json.loads(blob_data.decode('utf-8'))
        
        logger.info(f"Hiperparâmetros carregados: {blob_path}")
        return hyperparams
    
    except Exception as e:
        logger.exception(f"Erro ao carregar hiperparâmetros para {ticker}")
        raise

def load_history_data(container_client, ticker: str):
    """
    Carrega dados históricos do Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos
    """
    try:
        blob_path = f"history/{ticker}.parquet"
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise FileNotFoundError(f"Dados históricos não encontrados: {blob_path}")
        
        blob_data = blob_client.download_blob().readall()
        df = pd.read_parquet(io.BytesIO(blob_data))
        
        if df.empty:
            raise ValueError(f"Dados históricos vazios para {ticker}")
        
        logger.info(f"Dados históricos carregados: {blob_path} ({len(df)} registros)")
        return df
    
    except Exception as e:
        logger.exception(f"Erro ao carregar dados históricos para {ticker}")
        raise

def save_model(container_client, ticker: str, version: str, model_bytes: bytes, scaler_bytes: bytes):
    """
    Salva modelo e scaler no Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão do modelo (ex: "v1")
        model_bytes: Bytes do modelo PyTorch Lightning checkpoint
        scaler_bytes: Bytes do scaler (joblib pickle)
    
    Returns:
        dict: Caminhos dos arquivos salvos
    """
    try:
        model_path = f"models/{ticker}_v{version}.ckpt"
        scaler_path = f"models/{ticker}_v{version}_scaler.pkl"
        
        # Salvar modelo
        model_blob = container_client.get_blob_client(model_path)
        model_blob.upload_blob(model_bytes, overwrite=True)
        logger.info(f"Modelo salvo: {model_path}")
        
        # Salvar scaler
        scaler_blob = container_client.get_blob_client(scaler_path)
        scaler_blob.upload_blob(scaler_bytes, overwrite=True)
        logger.info(f"Scaler salvo: {scaler_path}")
        
        return {
            "model_path": model_path,
            "scaler_path": scaler_path
        }
    
    except Exception as e:
        logger.exception(f"Erro ao salvar modelo para {ticker}_v{version}")
        raise
