import os
import io
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import azure.functions as func
import pandas as pd
import yfinance as yf
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.function_name(name="fetch_day")
@app.route(route="fetch-day", methods=["GET"])
def fetch_day(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_day")
    
    logger.info("Iniciando execução da função fetch_day")
    
    tickers_env = os.getenv("TICKERS", "").strip()
    if not tickers_env:
        logger.error("Variável de ambiente TICKERS não definida.")
        return func.HttpResponse(
            "Erro: TICKERS não definido",
            status_code=400
        )

    container_name = os.getenv("BLOB_CONTAINER", "techchallenge04storage")
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        logger.error("AzureWebJobsStorage não definida.")
        return func.HttpResponse(
            "Erro: AzureWebJobsStorage não definido",
            status_code=500
        )

    # Data do dia no fuso de São Paulo (para partição year/month/day)
    tz = ZoneInfo("America/Sao_Paulo")
    now_sp = datetime.now(tz)
    year = f"{now_sp.year:04d}"
    month = f"{now_sp.month:02d}"
    day = f"{now_sp.day:02d}"

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)
        try:
            container_client.create_container()
            logger.info(f"Container {container_name} criado")
        except ResourceExistsError:
            logger.info(f"Container {container_name} já existe")
    except Exception as e:
        logger.exception("Falha ao conectar no Azure Blob Storage: %s", e)
        return func.HttpResponse(
            f"Erro ao conectar no storage: {str(e)}",
            status_code=500
        )

    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    logger.info("Processando tickers: %s", tickers)

    successful = 0
    failed = 0
    results = []

    for ticker in tickers:
        try:
            # Busca do dia (se não houver pregão, DataFrame pode vir vazio)
            df = yf.Ticker(f"{ticker}.SA").history(period="1d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                logger.warning("Sem dados para %s no dia %s-%s-%s", ticker, year, month, day)
                failed += 1
                results.append(f"{ticker}: Sem dados")
                continue

            # Garantir índice/colunas apropriados antes do parquet
            df = df.reset_index()  # 'Date' vira coluna
            
            # Adiciona metadata
            df["execution_timestamp"] = now_sp.isoformat()
            df["ticker"] = ticker

            # Serializa em Parquet para memória
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)

            # Caminho: year/month/day/{ticker}.parquet
            blob_path = f"{year}/{month}/{day}/{ticker}.parquet"
            blob_client = container_client.get_blob_client(blob_path)

            # Upload sobrescrevendo se já existir
            blob_client.upload_blob(buf, overwrite=True)
            logger.info("Gravado blob: %s", blob_path)
            successful += 1
            results.append(f"{ticker}: OK")

        except Exception as e:
            logger.exception("Falha ao processar %s: %s", ticker, e)
            failed += 1
            results.append(f"{ticker}: Erro - {str(e)}")

    # Resposta detalhada
    response_msg = {
        "status": "completed",
        "timestamp": now_sp.isoformat(),
        "successful": successful,
        "failed": failed,
        "results": results
    }
    
    logger.info(f"Execução finalizada: {successful} sucessos, {failed} falhas")
    
    return func.HttpResponse(
        body=str(response_msg),
        status_code=200,
        mimetype="application/json"
    )

@app.function_name(name="fetch_history")
@app.route(route="fetch-history", methods=["GET"])
def fetch_history(req: func.HttpRequest) -> func.HttpResponse:
    """
    Busca dados históricos dos tickers
    Salva em: history/{ticker}.parquet
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_history")

    logger.info("Iniciando execução da função fetch_history")

    # Pega configurações das env vars
    start_date = os.getenv("START_DATE", "2018-01-01")
    end_date = os.getenv("END_DATE", "2025-10-31")

    tickers_env = os.getenv("TICKERS", "").strip()
    if not tickers_env:
        logger.error("Variável de ambiente TICKERS não definida")
        return func.HttpResponse(
            body='{"success": false, "error": "TICKERS não definido"}',
            status_code=400,
            mimetype="application/json"
        )

    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    logger.info(f"Processando tickers: {tickers}")
    logger.info(f"Período: {start_date} até {end_date}")

    # Storage config
    container_name = os.getenv("BLOB_CONTAINER", "techchallenge04storage")
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        logger.error("AzureWebJobsStorage não definida")
        return func.HttpResponse(
            body='{"success": false, "error": "AzureWebJobsStorage não definido"}',
            status_code=500,
            mimetype="application/json"
        )

    # Conecta no storage
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)
        try:
            container_client.create_container()
            logger.info(f"Container {container_name} criado")
        except ResourceExistsError:
            logger.info(f"Container {container_name} já existe")
    except Exception as e:
        logger.exception("Falha ao conectar no Azure Blob Storage")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "Falha ao conectar no storage: {str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )

    # Timezone
    tz = ZoneInfo("America/Sao_Paulo")
    execution_time = datetime.now(tz)

    # Processa cada ticker
    try:
        for ticker in tickers:
            logger.info(f"Processando {ticker}...")

            # Busca dados históricos do yfinance
            ticker_obj = yf.Ticker(f"{ticker}.SA")
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False
            )

            if df is None or df.empty:
                logger.warning(f"Sem dados para {ticker}")
                continue

            logger.info(f"{ticker}: {len(df)} registros encontrados")

            # Reset index (Date vira coluna)
            df = df.reset_index()

            # Adiciona metadata
            df["execution_timestamp"] = execution_time.isoformat()
            df["ticker"] = ticker

            # Serializa em Parquet
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)

            # Upload no caminho history/{ticker}.parquet
            blob_path = f"history/{ticker}.parquet"
            blob_client = container_client.get_blob_client(blob_path)
            blob_client.upload_blob(buf, overwrite=True)

            logger.info(f"{ticker}: dados salvos em {blob_path}")

            # Pausa breve entre tickers (rate limiting)
            time.sleep(1)

        logger.info("Execução finalizada com sucesso")
        return func.HttpResponse(
            body='{"success": true}',
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logger.exception("Erro durante processamento")
        return func.HttpResponse(
            body=f'{{"success": false, "error": "{str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )
