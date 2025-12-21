from azure.cosmos import CosmosClient, PartitionKey, exceptions
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import json

logger = logging.getLogger(__name__)

def get_cosmos_client(conn_str: str, database_name: str):
    """
    Retorna Cosmos DB client e garante que database e containers existem
    
    Args:
        conn_str: Connection string do Cosmos DB
        database_name: Nome do database
    
    Returns:
        tuple: (CosmosClient, Database, Container model_versions, Container training_metrics)
    """
    try:
        client = CosmosClient.from_connection_string(conn_str)
        
        # Criar ou obter database
        try:
            database = client.create_database_if_not_exists(id=database_name)
            logger.info(f"Database {database_name} verificado/criado")
        except Exception as e:
            logger.exception(f"Erro ao criar/obter database {database_name}")
            raise
        
        # Criar ou obter container model_versions
        try:
            container_model_versions = database.create_container_if_not_exists(
                id="model_versions",
                partition_key=PartitionKey(path="/ticker"),
                offer_throughput=400
            )
            logger.info("Container model_versions verificado/criado")
        except Exception as e:
            logger.exception("Erro ao criar/obter container model_versions")
            raise
        
        # Criar ou obter container training_metrics
        try:
            container_training_metrics = database.create_container_if_not_exists(
                id="training_metrics",
                partition_key=PartitionKey(path="/ticker"),
                offer_throughput=400
            )
            logger.info("Container training_metrics verificado/criado")
        except Exception as e:
            logger.exception("Erro ao criar/obter container training_metrics")
            raise
        
        return client, database, container_model_versions, container_training_metrics
    
    except Exception as e:
        logger.exception("Falha ao conectar no Cosmos DB")
        raise

def get_next_version(container_model_versions, ticker: str):
    """
    Consulta última versão do modelo e retorna próxima versão
    
    Args:
        container_model_versions: Container do Cosmos DB
        ticker: Ticker da ação (ex: "PETR4")
    
    Returns:
        str: Próxima versão (ex: "v1", "v2", "v3")
    """
    try:
        # Query para buscar todas as versões do ticker ordenadas por versão
        query = f"SELECT * FROM c WHERE c.ticker = '{ticker}' ORDER BY c.version DESC"
        
        items = list(container_model_versions.query_items(
            query=query,
            partition_key=ticker
        ))
        
        if not items:
            # Primeira versão
            return "v1"
        
        # Pegar a última versão e incrementar
        last_version = items[0].get("version", "v0")
        # Remove "v" e converte para int
        version_num = int(last_version.replace("v", ""))
        next_version_num = version_num + 1
        
        return f"v{next_version_num}"
    
    except Exception as e:
        logger.exception(f"Erro ao obter próxima versão para {ticker}")
        # Em caso de erro, retorna v1 como fallback
        logger.warning(f"Retornando v1 como fallback para {ticker}")
        return "v1"

def get_latest_version(container_model_versions, ticker: str):
    """
    Retorna a versão mais recente do modelo para um ticker
    
    Args:
        container_model_versions: Container do Cosmos DB
        ticker: Ticker da ação (ex: "PETR4")
    
    Returns:
        str: Versão mais recente (ex: "v3") ou None se não houver versões
    """
    try:
        # Query para buscar todas as versões do ticker ordenadas por timestamp
        query = f"SELECT * FROM c WHERE c.ticker = '{ticker}' AND c.status = 'completed' ORDER BY c.timestamp DESC"
        
        items = list(container_model_versions.query_items(
            query=query,
            partition_key=ticker
        ))
        
        if not items:
            return None
        
        # Retornar a versão mais recente (primeiro item da lista ordenada)
        latest_version = items[0].get("version")
        logger.info(f"Versão mais recente encontrada para {ticker}: {latest_version}")
        return latest_version
    
    except Exception as e:
        logger.exception(f"Erro ao obter versão mais recente para {ticker}")
        return None

def save_model_version(container_model_versions, ticker: str, version: str, metrics: dict, 
                       hyperparams: dict, model_path: str, scaler_path: str, status: str = "completed"):
    """
    Salva registro de versão do modelo no Cosmos DB
    
    Args:
        container_model_versions: Container do Cosmos DB
        ticker: Ticker da ação
        version: Versão do modelo (ex: "v1")
        metrics: Dicionário com métricas de validação e teste
        hyperparams: Dicionário com todos os hiperparâmetros
        model_path: Caminho do modelo no Storage
        scaler_path: Caminho do scaler no Storage
        status: Status do treinamento ("completed", "failed", etc.)
    """
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        timestamp = datetime.now(tz).isoformat()
        
        document = {
            "id": f"{ticker}_{version}",
            "ticker": ticker,
            "version": version,
            "timestamp": timestamp,
            "status": status,
            "metrics": metrics,
            "hyperparams": hyperparams,  # Salva todos os hiperparâmetros
            "model_path": model_path,
            "scaler_path": scaler_path
        }
        
        container_model_versions.upsert_item(document)
        logger.info(f"Versão do modelo salva no Cosmos DB: {ticker}_{version}")
    
    except Exception as e:
        logger.exception(f"Erro ao salvar versão do modelo no Cosmos DB: {ticker}_{version}")
        raise

def save_training_metrics(container_training_metrics, ticker: str,
                            version: str, train_loss_history: list = None,
                            val_loss_history: list = None, learning_rates: list = None):
    """
    Salva métricas detalhadas de treinamento (opcional)
    
    Args:
        container_training_metrics: Container do Cosmos DB
        ticker: Ticker da ação
        version: Versão do modelo
        train_loss_history: Lista de loss de treinamento por epoch
        val_loss_history: Lista de loss de validação por epoch
        learning_rates: Lista de learning rates por epoch
    """
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        timestamp = datetime.now(tz).isoformat()
        
        document = {
            "id": f"{ticker}_{version}_{timestamp}",
            "ticker": ticker,
            "version": version,
            "timestamp": timestamp,
            "train_loss": train_loss_history or [],
            "val_loss": val_loss_history or [],
            "learning_rates": learning_rates or []
        }
        
        container_training_metrics.upsert_item(document)
        logger.info(f"Métricas de treinamento salvas no Cosmos DB: {ticker}_{version}")
    
    except Exception as e:
        logger.exception(f"Erro ao salvar métricas de treinamento no Cosmos DB: {ticker}_{version}")
        # Não levanta exceção pois é opcional
        logger.warning("Continuando sem salvar métricas detalhadas")
