"""Módulo para gerar especificação OpenAPI 3.0 para o Data Service"""

import os
from typing import Dict, Any


def _get_server_base() -> str:
    """
    Retorna o server base baseado no ambiente.
    - Local/Dev: /api
    - Azure (produção): /api/data (via APIM path-based routing)
    """
    # Verifica se está rodando no Azure (produção)
    # Azure Functions geralmente tem WEBSITE_INSTANCE_ID definido
    is_azure = os.getenv("WEBSITE_INSTANCE_ID") is not None
    
    if is_azure:
        return "/api/data"
    else:
        return "/api"


def get_openapi_spec() -> Dict[str, Any]:
    """
    Retorna a especificação OpenAPI 3.0 do Data Service
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "FIAP Tech Challenge 04 - Data Service API",
            "version": "1.0.0",
            "description": "API para coleta e processamento de dados de mercado financeiro. Busca dados históricos de ações, aplica feature engineering e armazena em Azure Blob Storage.",
            "contact": {
                "name": "Data Service API Support"
            }
        },
        "servers": [
            {
                "url": _get_server_base(),
                "description": "API base path (local: /api, Azure: /api/data via APIM)"
            }
        ],
        "tags": [
            {
                "name": "Health",
                "description": "Endpoints de verificação de saúde do serviço"
            },
            {
                "name": "Data Collection",
                "description": "Endpoints para coleta de dados de mercado"
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
                                        "service": "data-service",
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
                                        "service": "data-service",
                                        "error": "Connection timeout",
                                        "timestamp": "2024-01-15T10:30:00-03:00"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/fetch-day": {
                "get": {
                    "tags": ["Data Collection"],
                    "summary": "Buscar dados do dia",
                    "description": "Busca dados para os tickers configurados. Retorna os últimos N dias (configurável) de dados históricos, aplica feature engineering e salva no Azure Blob Storage.",
                    "operationId": "fetchDay",
                    "parameters": [
                        {
                            "name": "date",
                            "in": "query",
                            "description": "Data para buscar dados (formato: YYYY-MM-DD). Se não fornecido, usa a data atual. Não pode ser uma data futura.",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "format": "date",
                                "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                                "example": "2024-01-15"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Dados processados com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/FetchDayResponse"
                                    },
                                    "example": {
                                        "status": "completed",
                                        "target_date": "2024-01-15",
                                        "period": {
                                            "start": "2023-10-17",
                                            "end": "2024-01-15",
                                            "days": 90
                                        },
                                        "timestamp": "2024-01-15T10:30:00-03:00",
                                        "successful": 2,
                                        "failed": 0,
                                        "results": [
                                            "AAPL: OK (90 registros)",
                                            "MSFT: OK (90 registros)"
                                        ]
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Erro de validação (ex: formato de data inválido ou data futura)",
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
            "/fetch-history": {
                "get": {
                    "tags": ["Data Collection"],
                    "summary": "Buscar histórico completo",
                    "description": "Busca histórico completo dos tickers configurados. Aplica feature engineering em modo de treinamento e salva no Azure Blob Storage. Este endpoint busca todo o período histórico configurado (geralmente vários anos).",
                    "operationId": "fetchHistory",
                    "responses": {
                        "200": {
                            "description": "Histórico processado com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SuccessResponse"
                                    },
                                    "example": {
                                        "success": True
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Erro de configuração",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    },
                                    "example": {
                                        "success": False,
                                        "error": "Configuração inválida: tickers não definidos"
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
                                        "error": "Erro ao processar ticker AAPL: Connection timeout"
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
                            "example": "data-service",
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
                "FetchDayResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "example": "completed",
                            "description": "Status da operação"
                        },
                        "target_date": {
                            "type": "string",
                            "format": "date",
                            "example": "2024-01-15",
                            "description": "Data alvo processada"
                        },
                        "period": {
                            "type": "object",
                            "properties": {
                                "start": {
                                    "type": "string",
                                    "format": "date",
                                    "example": "2023-10-17",
                                    "description": "Data de início do período"
                                },
                                "end": {
                                    "type": "string",
                                    "format": "date",
                                    "example": "2024-01-15",
                                    "description": "Data de fim do período"
                                },
                                "days": {
                                    "type": "integer",
                                    "example": 90,
                                    "description": "Número de dias no período"
                                }
                            },
                            "required": ["start", "end", "days"]
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "example": "2024-01-15T10:30:00-03:00",
                            "description": "Timestamp da operação (ISO 8601)"
                        },
                        "successful": {
                            "type": "integer",
                            "example": 2,
                            "description": "Número de tickers processados com sucesso"
                        },
                        "failed": {
                            "type": "integer",
                            "example": 0,
                            "description": "Número de tickers que falharam"
                        },
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "example": ["AAPL: OK (90 registros)", "MSFT: OK (90 registros)"],
                            "description": "Lista de resultados detalhados por ticker"
                        }
                    },
                    "required": ["status", "target_date", "period", "timestamp", "successful", "failed", "results"]
                },
                "SuccessResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True,
                            "description": "Indica se a operação foi bem-sucedida"
                        }
                    },
                    "required": ["success"]
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
                            "example": "data-service"
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

