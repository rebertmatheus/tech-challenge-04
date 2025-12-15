import io
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

class ParquetHandler:
    """Manipula salvamento de dados em Parquet"""

    def __init__(self, container_client):
        self.container_client = container_client
        self.logger = logging.getLogger(__name__)
        self.tz = ZoneInfo("America/Sao_Paulo")

    def save_daily_data(self, df: pd.DataFrame, ticker: str, date: datetime):
        """Salva dados diários particionados por data"""

        df = df.copy()
        df["execution_timestamp"] = datetime.now(self.tz).isoformat()
        df["ticker"] = ticker

        year = f"{date.year:04d}"
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        blob_path = f"{year}/{month}/{day}/{ticker}.parquet"

        self._upload_parquet(df, blob_path)
        return blob_path

    def save_history_data(self, df: pd.DataFrame, ticker: str):
        """Salva histórico completo em history/{ticker}.parquet"""

        df = df.copy()
        df = df.reset_index()
        df["execution_timestamp"] = datetime.now(self.tz).isoformat()
        df["ticker"] = ticker

        blob_path = f"history/{ticker}.parquet"
        self._upload_parquet(df, blob_path)
        return blob_path

    def _upload_parquet(self, df: pd.DataFrame, blob_path: str):
        """Upload de DataFrame como Parquet"""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        blob_client = self.container_client.get_blob_client(blob_path)
        blob_client.upload_blob(buf, overwrite=True)

        self.logger.info(f"Blob salvo: {blob_path}")
