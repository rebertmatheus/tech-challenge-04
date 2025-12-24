import logging
import json
import azure.functions as func
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from utils.config import Config
from utils.storage import get_storage_client
from utils.yfinance_client import YFinanceClient
from utils.parquet_handler import ParquetHandler
from utils.feature_engineering import FeatureEngineer

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
            "service": "data-service",
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
            "service": "data-service",
            "error": str(e),
            "timestamp": now_sp.isoformat()
        }

        return func.HttpResponse(
            json.dumps(error_status),
            status_code=503,
            mimetype="application/json"
        )

@app.function_name(name="fetch_day")
@app.route(route="fetch-day", methods=["GET"])
def fetch_day(req: func.HttpRequest) -> func.HttpResponse:
    """Busca dados para os tickers configurados. Aceita parâmetro 'date' opcional (YYYY-MM-DD)"""
    logger = setup_logger("fetch_day")
    logger.info("Iniciando execução")
    
    engineer = FeatureEngineer()

    try:
        # Configuração
        tickers = Config.get_tickers()
        storage_config = Config.get_storage_config()

        if not storage_config["conn_str"]:
            return func.HttpResponse(
                body='{"success": false, "error": "AzureWebJobsStorage não definido"}',
                status_code=500,
                mimetype="application/json"
            )

        # Clientes
        container_client = get_storage_client(
            storage_config["conn_str"],
            storage_config["container"]
        )
        yf_client = YFinanceClient()
        parquet_handler = ParquetHandler(container_client)

        # Processar parâmetro 'date' (opcional)
        tz = ZoneInfo("America/Sao_Paulo")
        now_sp = datetime.now(tz)
        
        date_param = req.params.get('date')
        if date_param:
            try:
                # Validar e parsear data
                target_date = datetime.strptime(date_param, "%Y-%m-%d").date()
                # Converter para datetime com timezone para comparação
                target_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=tz)
                
                # Validar que não é data futura
                if target_datetime > now_sp:
                    return func.HttpResponse(
                        body=json.dumps({"success": False, "error": f"Data não pode ser futura: {date_param}"}),
                        status_code=400,
                        mimetype="application/json"
                    )
                
                logger.info(f"Data especificada: {target_date}")
            except ValueError:
                return func.HttpResponse(
                    body=json.dumps({"success": False, "error": f"Formato de data inválido: {date_param}. Use YYYY-MM-DD"}),
                    status_code=400,
                    mimetype="application/json"
                )
        else:
            # Se não fornecido, usar data atual
            target_date = now_sp.date()
            target_datetime = now_sp
            logger.info(f"Usando data atual: {target_date}")

        # Calcular período de busca (120 dias para trás)
        start_date = target_datetime - timedelta(days=120)
        end_date = target_datetime
        
        logger.info(f"Período de busca: {start_date.date()} até {end_date.date()} ({120} dias)")

        # Processa tickers
        successful = 0
        failed = 0
        results = []

        for ticker in tickers:
            try:
                df = yf_client.fetch_ticker_data(
                    ticker=ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )

                if df is None or df.empty:
                    failed += 1
                    results.append(f"{ticker}: Sem dados")
                    continue
                
                df = engineer.create_features(df, is_training_data=False)
                if df is None:
                    logger.warning(f"{ticker}: create_features retornou None")
                    continue
                
                logger.info(f"Features aplicadas: {df.shape}")

                # Salvar usando target_datetime (data especificada ou atual)
                blob_path = parquet_handler.save_daily_data(df, ticker, target_datetime)
                successful += 1
                results.append(f"{ticker}: OK ({len(df)} registros)")

            except Exception as e:
                logger.exception(f"Erro ao processar {ticker}")
                failed += 1
                results.append(f"{ticker}: Erro - {str(e)}")

        # Resposta
        response = {
            "status": "completed",
            "target_date": target_date.strftime("%Y-%m-%d"),
            "period": {
                "start": start_date.date().strftime("%Y-%m-%d"),
                "end": end_date.date().strftime("%Y-%m-%d"),
                "days": 120
            },
            "timestamp": target_datetime.isoformat(),
            "successful": successful,
            "failed": failed,
            "results": results
        }

        logger.info(f"Finalizado: {successful} sucessos, {failed} falhas. Período: {start_date.date()} até {end_date.date()}")

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )

    except ValueError as e:
        logger.error(f"Erro de configuração: {e}")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("Erro inesperado")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="fetch_history")
@app.route(route="fetch-history", methods=["GET"])
def fetch_history(req: func.HttpRequest) -> func.HttpResponse:
    """Busca histórico completo dos tickers"""
    logger = setup_logger("fetch_history")
    logger.info("Iniciando execução")
    
    engineer = FeatureEngineer()

    try:
        # Configuração
        tickers = Config.get_tickers()
        storage_config = Config.get_storage_config()
        date_range = Config.get_date_range()

        if not storage_config["conn_str"]:
            return func.HttpResponse(
                body='{"success": false, "error": "AzureWebJobsStorage não definido"}',
                status_code=500,
                mimetype="application/json"
            )

        logger.info(f"Período: {date_range['start']} até {date_range['end']}")

        # Clientes
        container_client = get_storage_client(
            storage_config["conn_str"],
            storage_config["container"]
        )
        yf_client = YFinanceClient(max_retries=3)
        parquet_handler = ParquetHandler(container_client)

        # Processa tickers
        for ticker in tickers:
            try:
                df = yf_client.fetch_ticker_data(
                    ticker=ticker,
                    start=date_range["start"],
                    end=date_range["end"]
                )

                if df is None or df.empty:
                    logger.warning(f"Sem dados para {ticker}")
                    continue
                
                logger.info("Aplicando feature engineering (training mode)...")
                df = engineer.create_features(df, is_training_data=True)
                
                if df is None:
                    logger.warning(f"{ticker}: create_features retornou None")
                    continue
                
                logger.info(f"Features criadas: {df.shape}")

                parquet_handler.save_history_data(df, ticker)
                logger.info(f"{ticker}: dados salvos com sucesso")

            except Exception as e:
                logger.exception(f"Erro ao processar {ticker}")
                raise

        logger.info("Execução finalizada com sucesso")
        return func.HttpResponse(
            body='{"success": true}',
            status_code=200,
            mimetype="application/json"
        )

    except ValueError as e:
        logger.error(f"Erro de configuração: {e}")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logger.exception("Erro inesperado")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )