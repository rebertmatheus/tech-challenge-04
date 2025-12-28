import os

class Config:
    """Configurações centralizadas"""
    
    @staticmethod
    def get_tickers():
        if tickers_env := os.getenv("TICKERS", "").strip():
            return [t.strip() for t in tickers_env.split(",") if t.strip()]
        else:
            raise ValueError("TICKERS não definido")

    @staticmethod
    def get_storage_config():
        return {
            "conn_str": os.getenv("AzureWebJobsStorage"),
            "container": os.getenv("BLOB_CONTAINER", "techchallenge04storage")
        }

    @staticmethod
    def get_date_range():
        return {
            "start": os.getenv("START_DATE", "2018-01-01"),
            "end": os.getenv("END_DATE", "2025-10-31")
        }
    
    @staticmethod
    def get_loopback_period():
        loopback_str = os.getenv("LOOPBACK_PERIOD", "120")
        try:
            return int(loopback_str)
        except ValueError:
            # Se não conseguir converter, retorna valor padrão
            return 120