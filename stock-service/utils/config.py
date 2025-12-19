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
    def get_cosmos_config():
        return {
            "conn_str": os.getenv("COSMOS_DB_CONNECTION_STRING"),
            "database": os.getenv("COSMOS_DB_DATABASE", "techchallenge04"),
            "container_model_versions": os.getenv("COSMOS_DB_CONTAINER_MODEL_VERSIONS", "model_versions"),
            "container_training_metrics": os.getenv("COSMOS_DB_CONTAINER_TRAINING_METRICS", "training_metrics")
        }

    @staticmethod
    def get_model_config():
        return {
            "default_version": "v1",
            "hyperparameters_path": "hyperparameters",
            "models_path": "models",
            "history_path": "history"
        }
