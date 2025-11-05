import os
import io
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import azure.functions as func
import pandas as pd
import yfinance as yf
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Timer: 19:00 de segunda a sexta (interpretação no fuso via WEBSITE_TIME_ZONE)
@app.function_name(name="fetch_yfinance")
@app.schedule(schedule="0 0 19 * * 1-5", arg_name="timer", use_monitor=True)
def fetch_yfinance(timer: func.TimerRequest) -> None:
    logger = logging.getLogger("fetch_yfinance")

    tickers_env = os.getenv("TICKERS", "").strip()
    if not tickers_env:
        logger.error("Variável de ambiente TICKERS não definida.")
        return

    container_name = os.getenv("BLOB_CONTAINER", "techchallenge04storage")
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        logger.error("AzureWebJobsStorage não definida.")
        return

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
        except ResourceExistsError:
            pass
    except Exception as e:
        logger.exception("Falha ao conectar no Azure Blob Storage: %s", e)
        return

    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    logger.info("Processando tickers: %s", tickers)

    for ticker in tickers:
        try:
            # Busca do dia (se não houver pregão, DataFrame pode vir vazio)
            df = yf.Ticker(ticker).history(period="1d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                logger.warning("Sem dados para %s no dia %s-%s-%s", ticker, year, month, day)
                continue

            # Garantir índice/colunas apropriados antes do parquet
            df = df.reset_index()  # 'Date' vira coluna
            # Opcional: normalizar tipos/colunas
            # df["Date"] = pd.to_datetime(df["Date"])

            # Serializa em Parquet para memória
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)

            # Caminho: year/month/day/{ticker}.parquet
            blob_path = f"{year}/{month}/{day}/{ticker}.parquet"
            blob_client = container_client.get_blob_client(blob_path)

            # Upload sobrescrevendo se já existir (ajuste se quiser evitar overwrite)
            blob_client.upload_blob(buf, overwrite=True)
            logger.info("Gravado blob: %s", blob_path)

        except Exception as e:
            logger.exception("Falha ao processar %s: %s", ticker, e)
