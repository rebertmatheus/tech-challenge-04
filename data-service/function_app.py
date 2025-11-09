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

@app.function_name(name="fetch_day")
@app.route(route="fetch-day", methods=["GET"])
def fetch_day(req: func.HttpRequest) -> func.HttpResponse:
    """Busca dados do dia atual para os tickers configurados"""
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

        # Data do dia
        tz = ZoneInfo("America/Sao_Paulo")
        now_sp = datetime.now(tz)
        
        start_date = now_sp - timedelta(days=90)

        # Processa tickers
        successful = 0
        failed = 0
        results = []

        for ticker in tickers:
            try:
                df = yf_client.fetch_ticker_data(
                    ticker=ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=now_sp.strftime("%Y-%m-%d")
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

                blob_path = parquet_handler.save_daily_data(df, ticker, now_sp)
                successful += 1
                results.append(f"{ticker}: OK")

            except Exception as e:
                logger.exception(f"Erro ao processar {ticker}")
                failed += 1
                results.append(f"{ticker}: Erro - {str(e)}")

        # Resposta
        response = {
            "status": "completed",
            "timestamp": now_sp.isoformat(),
            "successful": successful,
            "failed": failed,
            "results": results
        }

        logger.info(f"Finalizado: {successful} sucessos, {failed} falhas")

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