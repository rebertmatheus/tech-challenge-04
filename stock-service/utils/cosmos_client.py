from azure.cosmos import CosmosClient, PartitionKey, exceptions
import logging
from datetime import datetime, timedelta
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
        tuple: (CosmosClient, Database, Container model_versions, Container predictions)
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
        
        # Criar ou obter container predictions
        try:
            container_predictions = database.create_container_if_not_exists(
                id="predictions",
                partition_key=PartitionKey(path="/ticker"),
                offer_throughput=400
            )
            logger.info("Container predictions verificado/criado")
        except Exception as e:
            logger.exception("Erro ao criar/obter container predictions")
            raise
        
        return client, database, container_model_versions, container_predictions
    
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
                       hyperparams: dict, model_path: str, scaler_path: str, metrics_path: str, status: str = "completed"):
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
            "scaler_path": scaler_path,
            "metrics_path": metrics_path
        }
        
        container_model_versions.upsert_item(document)
        logger.info(f"Versão do modelo salva no Cosmos DB: {ticker}_{version}")
    
    except Exception as e:
        logger.exception(f"Erro ao salvar versão do modelo no Cosmos DB: {ticker}_{version}")
        raise

def save_prediction(container_predictions, ticker: str, model_version: str, 
                    prediction_date: str, predicted_price: float, data_date: str = None):
    """
    Salva predição no Cosmos DB (cache de predições)
    
    Args:
        container_predictions: Container do Cosmos DB para predições
        ticker: Ticker da ação (ex: "PETR4")
        model_version: Versão do modelo usado (ex: "v1")
        prediction_date: Data da predição (D+1, formato "YYYY-MM-DD")
        predicted_price: Preço predito
        data_date: Data dos dados usados para predição (opcional, formato "YYYY-MM-DD")
    """
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        timestamp = datetime.now(tz).isoformat()
        
        # ID único: ticker_model_version_prediction_date (inclui versão para permitir múltiplas versões)
        document = {
            "id": f"{ticker}_{model_version}_{prediction_date}",
            "ticker": ticker,
            "model_version": model_version,
            "prediction_date": prediction_date,  # Data predita (D+1)
            "data_date": data_date,  # Data dos dados usados (opcional)
            "predicted_price": predicted_price,
            "timestamp": timestamp
        }
        
        container_predictions.upsert_item(document)
        logger.info(f"Predição salva no Cosmos DB: {ticker}_{model_version} para {prediction_date}")
    
    except Exception as e:
        logger.exception(f"Erro ao salvar predição no Cosmos DB: {ticker}_{model_version}_{prediction_date}")
        raise

def get_prediction(container_predictions, ticker: str, model_version: str, prediction_date: str):
    """
    Busca predição em cache no Cosmos DB
    
    Args:
        container_predictions: Container do Cosmos DB para predições
        ticker: Ticker da ação (ex: "PETR4")
        model_version: Versão do modelo (ex: "v1")
        prediction_date: Data da predição (formato "YYYY-MM-DD")
    
    Returns:
        dict: Documento da predição ou None se não encontrado
    """
    try:
        item_id = f"{ticker}_{model_version}_{prediction_date}"
        prediction = container_predictions.read_item(
            item=item_id,
            partition_key=ticker
        )
        logger.info(f"Predição encontrada em cache: {ticker}_{model_version} para {prediction_date}")
        return prediction
    except exceptions.CosmosResourceNotFoundError:
        logger.info(f"Predição não encontrada em cache: {ticker}_{model_version} para {prediction_date}")
        return None
    except Exception as e:
        logger.exception(f"Erro ao buscar predição no Cosmos DB: {ticker}_{model_version}_{prediction_date}")
        raise