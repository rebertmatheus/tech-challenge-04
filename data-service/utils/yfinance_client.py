import yfinance as yf
import pandas as pd
import time
import logging
from typing import Optional

class YFinanceClient:
    """Cliente para buscar dados do yfinance com retry"""

    def __init__(self, max_retries: int = 3, backoff: float = 1.0):
        self.max_retries = max_retries
        self.backoff = backoff
        self.logger = logging.getLogger(__name__)

    def fetch_ticker_data(
        self, 
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Busca dados com retry automático"""

        for attempt in range(self.max_retries):
            try:
                ticker_obj = yf.Ticker(f"{ticker}.SA")
                df = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False
                )

                if df is None or df.empty:
                    self.logger.warning(f"Sem dados para {ticker}")
                    return None

                self.logger.info(f"{ticker}: {len(df)} registros encontrados")
                return df

            except Exception as e:
                self.logger.warning(
                    f"Tentativa {attempt + 1}/{self.max_retries} falhou para {ticker}: {e}"
                )

                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Falha ao buscar {ticker} após {self.max_retries} tentativas")
                    raise
        return None
