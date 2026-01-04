"""Módulo para gerar especificação OpenAPI 3.0 para o Stock Service"""

import os
from typing import Dict, Any


def _get_server_base() -> str:
    """
    Retorna o server base baseado no ambiente.
    - Local/Dev: /api
    - Azure (produção): /api/stock (via APIM path-based routing)
    """
    # Verifica se está rodando no Azure (produção)
    # Azure Functions geralmente tem WEBSITE_INSTANCE_ID definido
    is_azure = os.getenv("WEBSITE_INSTANCE_ID") is not None
    
    if is_azure:
        return "/api/stock"
    else:
        return "/api"


def get_openapi_spec() -> Dict[str, Any]:
    """
    Retorna a especificação OpenAPI 3.0 do Stock Service
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "FIAP Tech Challenge 04 - Stock Service API",
            "version": "1.0.0",
            "description": "API para treinamento de modelos LSTM e predição de preços de ações. Treina modelos de deep learning, armazena métricas no Cosmos DB e fornece predições D+1 para tickers configurados.",
            "contact": {
                "name": "Stock Service API Support"
            }
        },
        "servers": [
            {
                "url": _get_server_base(),
                "description": "API base path (local: /api, Azure: /api/stock via APIM)"
            }
        ],
        "tags": [
            {
                "name": "Health",
                "description": "Endpoints de verificação de saúde do serviço"
            },
            {
                "name": "Model Training",
                "description": "Endpoints para treinamento de modelos LSTM"
            },
            {
                "name": "Prediction",
                "description": "Endpoints para predição de preços"
            },
            {
                "name": "Metrics",
                "description": "Endpoints para consulta de métricas de modelos"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Health Check",
                    "description": "Verifica se o serviço está funcionando corretamente",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Serviço está funcionando",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    },
                                    "example": {
                                        "status": "healthy",
                                        "service": "stock-service",
                                        "timestamp": "2024-01-15T10:30:00-03:00"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Serviço está com problemas",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "status": "unhealthy",
                                        "service": "stock-service",
                                        "error": "Connection timeout",
                                        "timestamp": "2024-01-15T10:30:00-03:00"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/train": {
                "post": {
                    "tags": ["Model Training"],
                    "summary": "Treinar modelo LSTM",
                    "description": "Treina um modelo LSTM para um ticker específico. Carrega hiperparâmetros e dados históricos do Azure Blob Storage, executa o pipeline de treinamento e salva o modelo, scaler e métricas.",
                    "operationId": "train",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TrainRequest"
                                },
                                "example": {
                                    "ticker": "AAPL"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Treinamento concluído com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TrainResponse"
                                    },
                                    "example": {
                                        "success": True,
                                        "ticker": "AAPL",
                                        "version": "v1",
                                        "metrics": {
                                            "train_loss": 0.0012,
                                            "val_loss": 0.0015,
                                            "mse": 0.0015,
                                            "mae": 0.025,
                                            "rmse": 0.039
                                        },
                                        "model_path": "models/AAPL_v1.pt",
                                        "scaler_path": "models/AAPL_v1_scaler.pkl",
                                        "metrics_path": "models/AAPL_v1_metrics.json",
                                        "timestamp_start": "2024-01-15T10:00:00-03:00",
                                        "timestamp_end": "2024-01-15T10:30:00-03:00"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Erro de validação (ex: ticker não fornecido)",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Parâmetro 'ticker' é obrigatório"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Hiperparâmetros ou dados históricos não encontrados",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Hiperparâmetros não encontrados para AAPL"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Erro interno do servidor",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "AzureWebJobsStorage não definido"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "tags": ["Prediction"],
                    "summary": "Obter predição D+1",
                    "description": "Retorna a predição de preço para o próximo dia (D+1) de um ticker. Utiliza dados do dia especificado (ou atual) e retorna predição para D+1. Suporta cache via Cosmos DB e permite especificar versão do modelo.",
                    "operationId": "predict",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictRequest"
                                },
                                "example": {
                                    "ticker": "AAPL",
                                    "date": "2024-01-15",
                                    "version": "v1"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Predição realizada com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PredictResponse"
                                    },
                                    "example": {
                                        "success": True,
                                        "ticker": "AAPL",
                                        "model_version": "v1",
                                        "prediction_date": "2024-01-16",
                                        "predicted_price": 185.42,
                                        "prediction_timestamp": "2024-01-15T10:30:00-03:00",
                                        "from_cache": False
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Erro de validação (ex: ticker não fornecido, formato de data inválido)",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Formato de data inválido: 2024/01/15. Use YYYY-MM-DD"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Modelo, dados ou versão não encontrados",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Dados diários não encontrados para AAPL na data 2024-01-15"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Erro interno do servidor",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Erro ao executar predição: CUDA out of memory"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/metrics": {
                "post": {
                    "tags": ["Metrics"],
                    "summary": "Consultar métricas do modelo",
                    "description": "Retorna as métricas de treinamento de um modelo específico. Se a versão não for fornecida, retorna as métricas do modelo mais recente.",
                    "operationId": "metrics",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/MetricsRequest"
                                },
                                "example": {
                                    "ticker": "AAPL",
                                    "version": "v1"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Métricas retornadas com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MetricsResponse"
                                    },
                                    "example": {
                                        "success": True,
                                        "ticker": "AAPL",
                                        "version": "v1",
                                        "metrics": {
                                            "train_loss": 0.0012,
                                            "val_loss": 0.0015,
                                            "mse": 0.0015,
                                            "mae": 0.025,
                                            "rmse": 0.039
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Erro de validação (ex: ticker não fornecido)",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Parâmetro 'ticker' é obrigatório"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Métricas ou versão não encontradas",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Métricas não encontradas para AAPL_v1"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Erro interno do servidor",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "COSMOS_DB_CONNECTION_STRING não definido (necessário para buscar versão mais recente)"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "unhealthy"],
                            "description": "Status do serviço"
                        },
                        "service": {
                            "type": "string",
                            "example": "stock-service",
                            "description": "Nome do serviço"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Timestamp da verificação (ISO 8601)"
                        },
                        "error": {
                            "type": "string",
                            "description": "Mensagem de erro (apenas quando status é unhealthy)",
                            "example": "Connection timeout"
                        }
                    },
                    "required": ["status", "service", "timestamp"]
                },
                "TrainRequest": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker da ação (ex: AAPL, MSFT)",
                            "example": "AAPL"
                        }
                    },
                    "required": ["ticker"]
                },
                "TrainResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True,
                            "description": "Indica se o treinamento foi bem-sucedido"
                        },
                        "ticker": {
                            "type": "string",
                            "example": "AAPL",
                            "description": "Ticker processado"
                        },
                        "version": {
                            "type": "string",
                            "example": "v1",
                            "description": "Versão do modelo treinado"
                        },
                        "metrics": {
                            "type": "object",
                            "description": "Métricas de treinamento",
                            "properties": {
                                "train_loss": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0012,
                                    "description": "Loss de treinamento"
                                },
                                "val_loss": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0015,
                                    "description": "Loss de validação"
                                },
                                "mse": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0015,
                                    "description": "Mean Squared Error"
                                },
                                "mae": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.025,
                                    "description": "Mean Absolute Error"
                                },
                                "rmse": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.039,
                                    "description": "Root Mean Squared Error"
                                }
                            }
                        },
                        "model_path": {
                            "type": "string",
                            "example": "models/AAPL_v1.pt",
                            "description": "Caminho do modelo salvo no Azure Blob Storage"
                        },
                        "scaler_path": {
                            "type": "string",
                            "example": "models/AAPL_v1_scaler.pkl",
                            "description": "Caminho do scaler salvo no Azure Blob Storage"
                        },
                        "metrics_path": {
                            "type": "string",
                            "example": "models/AAPL_v1_metrics.json",
                            "description": "Caminho das métricas salvas no Azure Blob Storage"
                        },
                        "timestamp_start": {
                            "type": "string",
                            "format": "date-time",
                            "example": "2024-01-15T10:00:00-03:00",
                            "description": "Timestamp de início do treinamento (ISO 8601)"
                        },
                        "timestamp_end": {
                            "type": "string",
                            "format": "date-time",
                            "example": "2024-01-15T10:30:00-03:00",
                            "description": "Timestamp de fim do treinamento (ISO 8601)"
                        }
                    },
                    "required": ["success", "ticker", "version", "metrics", "model_path", "scaler_path", "metrics_path", "timestamp_start", "timestamp_end"]
                },
                "PredictRequest": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker da ação (ex: AAPL, MSFT)",
                            "example": "AAPL"
                        },
                        "date": {
                            "type": "string",
                            "format": "date",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                            "description": "Data base para predição (formato: YYYY-MM-DD). Se não fornecido, usa a data atual. A predição será para D+1 (próximo dia).",
                            "example": "2024-01-15"
                        },
                        "version": {
                            "type": "string",
                            "description": "Versão do modelo a usar. Se não fornecido, usa a versão mais recente.",
                            "example": "v1"
                        }
                    },
                    "required": ["ticker"]
                },
                "PredictResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True,
                            "description": "Indica se a predição foi bem-sucedida"
                        },
                        "ticker": {
                            "type": "string",
                            "example": "AAPL",
                            "description": "Ticker processado"
                        },
                        "model_version": {
                            "type": "string",
                            "example": "v1",
                            "description": "Versão do modelo utilizado"
                        },
                        "prediction_date": {
                            "type": "string",
                            "format": "date",
                            "example": "2024-01-16",
                            "description": "Data da predição (D+1, próximo dia)"
                        },
                        "predicted_price": {
                            "type": "number",
                            "format": "float",
                            "example": 185.42,
                            "description": "Preço predito para o próximo dia (D+1)"
                        },
                        "prediction_timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "example": "2024-01-15T10:30:00-03:00",
                            "description": "Timestamp da predição (ISO 8601)"
                        },
                        "from_cache": {
                            "type": "boolean",
                            "example": False,
                            "description": "Indica se a predição foi retornada do cache (Cosmos DB)"
                        }
                    },
                    "required": ["success", "ticker", "model_version", "prediction_date", "predicted_price", "prediction_timestamp", "from_cache"]
                },
                "MetricsRequest": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker da ação (ex: AAPL, MSFT)",
                            "example": "AAPL"
                        },
                        "version": {
                            "type": "string",
                            "description": "Versão do modelo. Se não fornecido, retorna métricas da versão mais recente.",
                            "example": "v1"
                        }
                    },
                    "required": ["ticker"]
                },
                "MetricsResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True,
                            "description": "Indica se a consulta foi bem-sucedida"
                        },
                        "ticker": {
                            "type": "string",
                            "example": "AAPL",
                            "description": "Ticker consultado"
                        },
                        "version": {
                            "type": "string",
                            "example": "v1",
                            "description": "Versão do modelo consultado"
                        },
                        "metrics": {
                            "type": "object",
                            "description": "Métricas de treinamento do modelo",
                            "properties": {
                                "train_loss": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0012,
                                    "description": "Loss de treinamento"
                                },
                                "val_loss": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0015,
                                    "description": "Loss de validação"
                                },
                                "mse": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0015,
                                    "description": "Mean Squared Error"
                                },
                                "mae": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.025,
                                    "description": "Mean Absolute Error"
                                },
                                "rmse": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.039,
                                    "description": "Root Mean Squared Error"
                                }
                            }
                        }
                    },
                    "required": ["success", "ticker", "version", "metrics"]
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": False,
                            "description": "Indica se a operação falhou"
                        },
                        "error": {
                            "type": "string",
                            "example": "Erro ao processar dados",
                            "description": "Mensagem de erro descritiva"
                        },
                        "status": {
                            "type": "string",
                            "description": "Status do serviço (apenas em respostas de health)",
                            "example": "unhealthy"
                        },
                        "service": {
                            "type": "string",
                            "description": "Nome do serviço (apenas em respostas de health)",
                            "example": "stock-service"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Timestamp do erro (apenas em respostas de health)",
                            "example": "2024-01-15T10:30:00-03:00"
                        }
                    },
                    "required": ["success", "error"]
                }
            }
        }
    }

