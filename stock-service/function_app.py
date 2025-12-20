import logging
import json
import azure.functions as func
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.config import Config
from utils.storage import get_storage_client, load_hyperparameters, load_history_data, save_model
from utils.cosmos_client import get_cosmos_client, get_next_version, save_model_version
from utils.trainer import ModelTrainer
from utils.model import StocksLSTM

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def setup_logger(name: str):
    """Configura logger estruturado"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)

@app.function_name(name="health_check")
@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Verifica se o serviço está funcionando"""
    try:
        # Data do dia
        tz = ZoneInfo("America/Sao_Paulo")
        now_sp = datetime.now(tz)
        
        health_status = {
            "status": "healthy",
            "service": "stock-service",
            "timestamp": now_sp.isoformat()
        }

        return func.HttpResponse(
            json.dumps(health_status),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "service": "stock-service",
            "error": str(e),
            "timestamp": now_sp.isoformat()
        }

        return func.HttpResponse(
            json.dumps(error_status),
            status_code=503,
            mimetype="application/json"
        )

@app.function_name(name="train")
@app.route(route="train", methods=["POST"])
def train(req: func.HttpRequest) -> func.HttpResponse:
    """Treina modelo LSTM para os tickers configurados"""
    logger = setup_logger("train")
    logger.info("Iniciando execução do endpoint /train")
    
    tz = ZoneInfo("America/Sao_Paulo")
    timestamp_start = datetime.now(tz).isoformat()
    
    try:
        # 1. Validar e obter parâmetros
        ticker  = (json.loads(req.get_body().decode()) if req.get_body() else {}).get('ticker')
        
        if not ticker:
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": "Parâmetro 'ticker' é obrigatório"}),
                status_code=400,
                mimetype="application/json"
            )
        
        ticker = ticker.strip().upper()
        logger.info(f"Treinando modelo para ticker: {ticker}")
        
        # 2. Carregar configurações
        logger.info("Carregando configurações...")
        storage_config = Config.get_storage_config()
        cosmos_config = Config.get_cosmos_config()
        
        if not storage_config["conn_str"]:
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": "AzureWebJobsStorage não definido"}),
                status_code=500,
                mimetype="application/json"
            )
        
        if not cosmos_config["conn_str"]:
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": "COSMOS_DB_CONNECTION_STRING não definido"}),
                status_code=500,
                mimetype="application/json"
            )
        
        # 3. Conectar aos serviços
        logger.info("Conectando aos serviços Azure...")
        container_client = get_storage_client(storage_config["conn_str"], storage_config["container"])
        _, _, container_model_versions, container_training_metrics = get_cosmos_client(
            cosmos_config["conn_str"], 
            cosmos_config["database"]
        )
        
        # 4. Carregar hiperparâmetros
        logger.info(f"Carregando hiperparâmetros para {ticker}...")
        try:
            hyperparams = load_hyperparameters(container_client, ticker)
        except FileNotFoundError as e:
            logger.error(f"Hiperparâmetros não encontrados: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Hiperparâmetros não encontrados para {ticker}"}),
                status_code=404,
                mimetype="application/json"
            )
        
        # 5. Carregar dados históricos
        logger.info(f"Carregando dados históricos para {ticker}...")
        try:
            df = load_history_data(container_client, ticker)
        except FileNotFoundError as e:
            logger.error(f"Dados históricos não encontrados: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Dados históricos não encontrados para {ticker}"}),
                status_code=404,
                mimetype="application/json"
            )
        except ValueError as e:
            logger.error(f"Dados históricos inválidos: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Dados históricos inválidos para {ticker}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # 6. Gerenciar versionamento
        logger.info("Gerenciando versionamento do modelo...")
        model_version = get_next_version(container_model_versions, ticker)
        logger.info(f"Próxima versão do modelo: {model_version}")
        
        # 7. Instânciar o Modelo e executar pipeline de treinamento
        logger.info("Instanciando modelo LSTM...")
        model = StocksLSTM(hyperparams)
        logger.info("Iniciando pipeline de treinamento...")
        trainer = ModelTrainer()
        model_bytes, scaler_bytes, metrics = trainer.train(ticker, hyperparams, df)
        logger.info("Treinamento concluído com sucesso")
        
        # 8. Salvar artefatos no Storage
        logger.info(f"Salvando modelo e scaler no Storage...")
        paths = save_model(container_client, ticker, model_version, model_bytes, scaler_bytes)
        logger.info(f"Artefatos salvos: {paths}")
        
        # 9. Salvar métricas no Cosmos DB
        logger.info("Salvando métricas no Cosmos DB...")
        save_model_version(
            container_model_versions,
            ticker=ticker,
            version=model_version,
            metrics=metrics,
            hyperparams=hyperparams,
            model_path=paths["model_path"],
            scaler_path=paths["scaler_path"],
            status="completed"
        )
        logger.info("Métricas salvas no Cosmos DB")
        
        # 10. Resposta de sucesso
        timestamp_end = datetime.now(tz).isoformat()
        response = {
            "success": True,
            "ticker": ticker,
            "version": model_version,
            "metrics": metrics,
            "model_path": paths["model_path"],
            "scaler_path": paths["scaler_path"],
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end
        }
        
        logger.info(f"Treinamento concluído com sucesso para {ticker}_{model_version}")
        
        return func.HttpResponse(
            body=json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )
    
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=404,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("Erro inesperado durante treinamento")
        
        # Tentar marcar versão como failed no Cosmos DB se possível
        try:
            if 'model_version' in locals() and 'ticker' in locals():
                save_model_version(
                    container_model_versions,
                    ticker=ticker,
                    version=model_version,
                    metrics={},
                    hyperparams={},
                    model_path="",
                    scaler_path="",
                    status="failed"
                )
        except:
            pass  # Ignorar erro ao tentar salvar status failed
        
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="predict")
@app.route(route="predict", methods=["GET", "POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Retorna predição D+1 para os tickers configurados"""
    print("Hello")
    logger = setup_logger("predict")
    logger.info("Iniciando execução")
    
    try:
        # TODO: Implementar lógica de predição
        response = {
            "success": True,
            "message": "Predict endpoint - Hello"
        }

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("Erro inesperado")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )