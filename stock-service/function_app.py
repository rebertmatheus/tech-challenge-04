import logging
import json
import azure.functions as func
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from utils.config import Config
from utils.storage import get_storage_client, load_hyperparameters, load_history_data, save_model, load_metrics, load_model, load_scaler, load_daily_data
from utils.cosmos_client import get_cosmos_client, get_next_version, save_model_version, get_latest_version, save_prediction, get_prediction
from utils.trainer import ModelTrainer
from utils.stocks_lstm import StocksLSTM
from utils.predictor import prepare_prediction_sequence, load_model_from_bytes, predict_price

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
    
    # Data do dia
    tz = ZoneInfo("America/Sao_Paulo")
    now_sp = datetime.now(tz)
    
    try:        
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
        _, _, container_model_versions, _ = get_cosmos_client(
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
        model_bytes, scaler_bytes, metrics_bytes, metrics = trainer.train(model, ticker, hyperparams, df)
        logger.info("Treinamento concluído com sucesso")
        
        # 8. Salvar artefatos no Storage
        logger.info(f"Salvando modelo, scaler e métricas no Storage...")
        paths = save_model(container_client, ticker, model_version, model_bytes, scaler_bytes, metrics_bytes)
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
            metrics_path=paths["metrics_path"],
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
            "metrics_path": paths["metrics_path"],
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
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Retorna predição D+1 para um ticker"""
    logger = setup_logger("predict")
    logger.info("Iniciando execução do endpoint /predict")
    
    tz = ZoneInfo("America/Sao_Paulo")
    prediction_timestamp = datetime.now(tz).isoformat()
    
    try:
        # 1. Validar e obter parâmetros do body
        body = json.loads(req.get_body().decode()) if req.get_body() else {}
        ticker = body.get('ticker')
        date_str = body.get('date')  # Opcional: "YYYY-MM-DD"
        model_version = body.get('version')  # Opcional: versão do modelo
        
        if not ticker:
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": "Parâmetro 'ticker' é obrigatório"}),
                status_code=400,
                mimetype="application/json"
            )
        
        ticker = ticker.strip().upper()
        
        # Processar data: se não fornecida, usar data atual
        if date_str:
            try:
                prediction_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": f"Formato de data inválido: {date_str}. Use YYYY-MM-DD"}),
                    status_code=400,
                    mimetype="application/json"
                )
        else:
            prediction_date = datetime.now(tz).date()
        
        logger.info(f"Predição para ticker: {ticker}, data: {prediction_date}, versão: {model_version or 'mais recente'}")
        
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
        
        # 3. Conectar aos serviços
        logger.info("Conectando aos serviços Azure...")
        container_client = get_storage_client(storage_config["conn_str"], storage_config["container"])
        
        # 4. Calcular data predita (D+1)
        # Se usamos dados do dia 26, estamos predizendo o dia 27
        predicted_date = prediction_date + timedelta(days=1)
        predicted_date_str = predicted_date.strftime("%Y-%m-%d")
        data_date_str = prediction_date.strftime("%Y-%m-%d")
        
        logger.info(f"Usando dados do dia {data_date_str} para predizer o dia {predicted_date_str}")
        
        # 5. Conectar ao Cosmos DB (se disponível)
        cosmos_connected = False
        container_model_versions = None
        container_predictions = None
        
        if cosmos_config["conn_str"]:
            try:
                _, _, container_model_versions, container_predictions = get_cosmos_client(
                    cosmos_config["conn_str"], 
                    cosmos_config["database"]
                )
                cosmos_connected = True
                logger.info("Conectado ao Cosmos DB")
            except Exception as e:
                logger.warning(f"Erro ao conectar ao Cosmos DB: {e}. Continuando sem cache...")
        
        # 6. Determinar versão do modelo a usar (precisa ser antes de verificar cache)
        if not model_version:
            if not cosmos_connected or not container_model_versions:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": "COSMOS_DB_CONNECTION_STRING não definido (necessário para buscar versão mais recente)"}),
                    status_code=500,
                    mimetype="application/json"
                )
            
            logger.info("Buscando versão mais recente no Cosmos DB...")
            model_version = get_latest_version(container_model_versions, ticker)
            
            if not model_version:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": f"Nenhuma versão de modelo encontrada para {ticker}"}),
                    status_code=404,
                    mimetype="application/json"
                )
            
            logger.info(f"Versão mais recente encontrada: {model_version}")
        else:
            model_version = model_version.strip()
        
        # 7. Verificar cache de predição no Cosmos DB (com versão específica)
        if cosmos_connected and container_predictions:
            try:
                cached_prediction = get_prediction(container_predictions, ticker, model_version, predicted_date_str)
                if cached_prediction:
                    logger.info(f"Predição encontrada em cache para {ticker}_{model_version} no dia {predicted_date_str}")
                    response = {
                        "success": True,
                        "ticker": ticker,
                        "model_version": model_version,
                        "prediction_date": predicted_date_str,
                        "predicted_price": cached_prediction["predicted_price"],
                        "prediction_timestamp": cached_prediction.get("timestamp", prediction_timestamp),
                        "from_cache": True
                    }
                    return func.HttpResponse(
                        body=json.dumps(response, indent=2),
                        status_code=200,
                        mimetype="application/json"
                    )
            except Exception as e:
                logger.warning(f"Erro ao verificar cache: {e}. Continuando com predição...")
        
        # 8. Carregar hiperparâmetros
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
        
        # 9. Carregar modelo e scaler
        logger.info(f"Carregando modelo e scaler para {ticker}_{model_version}...")
        try:
            model_bytes = load_model(container_client, ticker, model_version)
            scalers_dict = load_scaler(container_client, ticker, model_version)
            feature_scaler = scalers_dict['feature_scaler']
            target_scaler = scalers_dict['target_scaler']
        except FileNotFoundError as e:
            logger.error(f"Modelo ou scaler não encontrados: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Modelo ou scaler não encontrados para {ticker}_{model_version}"}),
                status_code=404,
                mimetype="application/json"
            )
        
        # 10. Carregar dados diários para predição
        logger.info(f"Carregando dados diários para {ticker} na data {data_date_str}...")
        try:
            df = load_daily_data(container_client, ticker, prediction_date)
        except FileNotFoundError as e:
            logger.error(f"Dados diários não encontrados: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Dados diários não encontrados para {ticker} na data {data_date_str}"}),
                status_code=404,
                mimetype="application/json"
            )
        except ValueError as e:
            logger.error(f"Dados diários inválidos: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Dados diários inválidos para {ticker} na data {data_date_str}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # 11. Preparar dados (aplicar DROP_COLUMNS se houver)
        logger.info("Preparando dados para predição...")
        drop_columns = hyperparams.get("DROP_COLUMNS", [])
        df_clean = df.drop(columns=drop_columns, errors='ignore')
        
        # 12. Preparar sequência temporal
        logger.info("Preparando sequência temporal...")
        try:
            sequence = prepare_prediction_sequence(df_clean, hyperparams, feature_scaler)
        except ValueError as e:
            logger.error(f"Erro ao preparar sequência: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": str(e)}),
                status_code=400,
                mimetype="application/json"
            )
        
        # 13. Carregar modelo e executar predição
        logger.info("Carregando modelo e executando predição...")
        try:
            model = load_model_from_bytes(model_bytes, hyperparams)
            predicted_price = predict_price(model, sequence, target_scaler)
        except Exception as e:
            logger.exception("Erro ao executar predição")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Erro ao executar predição: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )
        
        # 14. Salvar predição no Cosmos DB (cache)
        if cosmos_connected and container_predictions:
            try:
                save_prediction(
                    container_predictions,
                    ticker=ticker,
                    model_version=model_version,
                    prediction_date=predicted_date_str,
                    predicted_price=predicted_price,
                    data_date=data_date_str
                )
                logger.info(f"Predição salva no Cosmos DB para {ticker} no dia {predicted_date_str}")
            except Exception as e:
                logger.warning(f"Erro ao salvar predição no Cosmos DB: {e}. Continuando...")
        
        # 15. Resposta de sucesso
        response = {
            "success": True,
            "ticker": ticker,
            "model_version": model_version,
            "prediction_date": predicted_date_str,  # Data predita (D+1)
            "predicted_price": round(predicted_price, 2),
            "prediction_timestamp": prediction_timestamp,
            "from_cache": False
        }
        
        logger.info(f"Predição concluída com sucesso para {ticker}_{model_version}: R$ {predicted_price:.2f} (predição para {predicted_date_str})")
        
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
        logger.exception("Erro inesperado durante predição")
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="metrics")
@app.route(route="metrics", methods=["POST"])
def metrics(req: func.HttpRequest) -> func.HttpResponse:
    """Retorna métricas de um modelo treinado"""
    logger = setup_logger("metrics")
    logger.info("Iniciando execução do endpoint /metrics")
    
    try:
        # 1. Validar e obter parâmetros do body
        body = json.loads(req.get_body().decode()) if req.get_body() else {}
        ticker = body.get('ticker')
        version = body.get('version')  # Opcional
        
        if not ticker:
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": "Parâmetro 'ticker' é obrigatório"}),
                status_code=400,
                mimetype="application/json"
            )
        
        ticker = ticker.strip().upper()
        logger.info(f"Consultando métricas para ticker: {ticker}, versão: {version or 'mais recente'}")
        
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
        
        # 3. Conectar aos serviços
        logger.info("Conectando aos serviços Azure...")
        container_client = get_storage_client(storage_config["conn_str"], storage_config["container"])
        
        # 4. Determinar versão a usar
        if not version:
            # Buscar versão mais recente no Cosmos DB
            if not cosmos_config["conn_str"]:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": "COSMOS_DB_CONNECTION_STRING não definido (necessário para buscar versão mais recente)"}),
                    status_code=500,
                    mimetype="application/json"
                )
            
            _, _, container_model_versions, _ = get_cosmos_client(
                cosmos_config["conn_str"], 
                cosmos_config["database"]
            )
            
            logger.info("Buscando versão mais recente no Cosmos DB...")
            version = get_latest_version(container_model_versions, ticker)
            
            if not version:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": f"Nenhuma versão encontrada para {ticker}"}),
                    status_code=404,
                    mimetype="application/json"
                )
            
            logger.info(f"Versão mais recente encontrada: {version}")
        else:
            version = version.strip()
        
        # 5. Carregar métricas do Storage
        logger.info(f"Carregando métricas para {ticker}_{version}...")
        try:
            metrics_data = load_metrics(container_client, ticker, version)
        except FileNotFoundError as e:
            logger.error(f"Métricas não encontradas: {e}")
            return func.HttpResponse(
                body=json.dumps({"success": False, "error": f"Métricas não encontradas para {ticker}_{version}"}),
                status_code=404,
                mimetype="application/json"
            )
        
        # 6. Resposta de sucesso
        response = {
            "success": True,
            "ticker": ticker,
            "version": version,
            "metrics": metrics_data
        }
        
        logger.info(f"Métricas retornadas com sucesso para {ticker}_{version}")
        
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
        logger.error(f"Métricas não encontradas: {e}")
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=404,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("Erro inesperado ao consultar métricas")
        return func.HttpResponse(
            body=json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )