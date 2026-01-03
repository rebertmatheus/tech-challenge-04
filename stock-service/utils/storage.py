from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import logging
import json
import pandas as pd
import io
import joblib
import pandas as pd

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

def load_hyperparameters(container_client, ticker: str):
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
        blob_path = f"hyperparameters/{ticker}.json"
        
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
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

def save_model(container_client, ticker: str, version: str, model_bytes: bytes, scaler_bytes: bytes, metrics_bytes: bytes = None):
    """
    Salva modelo, scaler e métricas no Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão do modelo (ex: "v1")
        model_bytes: Bytes do modelo PyTorch Lightning checkpoint
        scaler_bytes: Bytes do scaler (joblib pickle)
        metrics_bytes: Bytes das métricas (joblib pickle, opcional)
    
    Returns:
        dict: Caminhos dos arquivos salvos
    """
    try:
        model_path = f"models/{ticker}_{version}.ckpt"
        scaler_path = f"models/{ticker}_{version}_scaler.pkl"
        
        # Salvar modelo
        model_blob = container_client.get_blob_client(model_path)
        model_blob.upload_blob(model_bytes, overwrite=True)
        logger.info(f"Modelo salvo: {model_path}")
        
        # Salvar scaler
        scaler_blob = container_client.get_blob_client(scaler_path)
        scaler_blob.upload_blob(scaler_bytes, overwrite=True)
        logger.info(f"Scaler salvo: {scaler_path}")
        
        result = {
            "model_path": model_path,
            "scaler_path": scaler_path
        }
        
        # Salvar métricas se fornecidas
        if metrics_bytes:
            metrics_path = f"models/{ticker}_{version}_metrics.pkl"
            metrics_blob = container_client.get_blob_client(metrics_path)
            metrics_blob.upload_blob(metrics_bytes, overwrite=True)
            logger.info(f"Métricas salvas: {metrics_path}")
            result["metrics_path"] = metrics_path
        
        return result
    
    except Exception as e:
        logger.exception(f"Erro ao salvar modelo para {ticker}_{version}")
        raise

def load_metrics(container_client, ticker: str, version: str):
    """
    Carrega métricas do Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão do modelo (ex: "v1")
    
    Returns:
        dict: Dicionário com métricas (validacao, teste, config)
    """
    try:
        blob_path = f"models/{ticker}_{version}_metrics.pkl"
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise FileNotFoundError(f"Métricas não encontradas: {blob_path}")
        
        blob_data = blob_client.download_blob().readall()
        metrics = joblib.load(io.BytesIO(blob_data))
        
        logger.info(f"Métricas carregadas: {blob_path}")
        return metrics
    
    except Exception as e:
        logger.exception(f"Erro ao carregar métricas para {ticker}_{version}")
        raise

def load_model(container_client, ticker: str, version: str):
    """
    Carrega modelo do Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão do modelo (ex: "v1")
    
    Returns:
        bytes: Bytes do modelo checkpoint
    """
    try:
        blob_path = f"models/{ticker}_{version}.ckpt"
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {blob_path}")
        
        blob_data = blob_client.download_blob().readall()
        
        logger.info(f"Modelo carregado: {blob_path} ({len(blob_data)} bytes)")
        return blob_data
    
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo para {ticker}_{version}")
        raise

def load_scaler(container_client, ticker: str, version: str):
    """
    Carrega scaler do Azure Blob Storage
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        version: Versão do modelo (ex: "v1")
    
    Returns:
        dict: Dicionário com 'feature_scaler' e 'target_scaler'
    """
    try:
        blob_path = f"models/{ticker}_{version}_scaler.pkl"
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise FileNotFoundError(f"Scaler não encontrado: {blob_path}")
        
        blob_data = blob_client.download_blob().readall()
        scalers = joblib.load(io.BytesIO(blob_data))
        
        logger.info(f"Scaler carregado: {blob_path}")
        return scalers
    
    except Exception as e:
        logger.exception(f"Erro ao carregar scaler para {ticker}_{version}")
        raise

def load_daily_data(container_client, ticker: str, date):
    """
    Carrega dados diários do Azure Blob Storage para uma data específica
    
    Args:
        container_client: Container client do Azure Blob Storage
        ticker: Ticker da ação (ex: "PETR4")
        date: Objeto datetime ou string no formato "YYYY-MM-DD"
    
    Returns:
        pd.DataFrame: DataFrame com dados do dia
    """
    try:
        # Converter date para datetime se for string
        if isinstance(date, str):
            from datetime import datetime
            date = datetime.strptime(date, "%Y-%m-%d")
        
        # Formatar caminho: YYYY/MM/DD/ticker.parquet
        year = f"{date.year:04d}"
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        blob_path = f"{year}/{month}/{day}/{ticker}.parquet"
        
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise FileNotFoundError(f"Dados diários não encontrados: {blob_path}")
        
        blob_data = blob_client.download_blob().readall()
        df = pd.read_parquet(io.BytesIO(blob_data))
        
        if df.empty:
            raise ValueError(f"Dados diários vazios para {ticker} na data {date.strftime('%Y-%m-%d')}")
        
        logger.info(f"Dados diários carregados: {blob_path} ({len(df)} registros)")
        return df
    
    except Exception as e:
        logger.exception(f"Erro ao carregar dados diários para {ticker} na data {date}")
        raise
