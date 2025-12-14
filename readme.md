# Tech Challenge 04 - Sistema de Predição de Ações com LSTM

Sistema completo de predição de preços de ações da B3 utilizando redes neurais LSTM (Long Short-Term Memory), implementado na nuvem Azure com arquitetura serverless.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Status do Projeto](#status-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Setup](#instalação-e-setup)
- [Documentação Detalhada](#documentação-detalhada)
- [Roadmap](#roadmap)

---

## 🎯 Visão Geral

Este projeto implementa um sistema end-to-end para predição de preços de ações utilizando:

- **Coleta Automática de Dados**: Azure Functions que buscam dados históricos e diários via yfinance
- **Feature Engineering**: 23 indicadores técnicos otimizados para predição de curto prazo
- **Modelo LSTM**: Rede neural recorrente com PyTorch Lightning para previsão de preço ajustado (D+1)
- **Infraestrutura Cloud**: Azure Functions, Storage Account, Cosmos DB, API Management
- **Monitoramento**: Application Insights e dashboards customizados

### Objetivo

Prever o preço de fechamento ajustado (Adj Close) de ações da B3 para o próximo dia útil (D+1) utilizando dados históricos e indicadores técnicos.

---

## 🏗️ Arquitetura

```
┌─────────────────┐
│   Logic App     │  (Trigger diário)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data-Service   │  (Azure Function)
│  - fetch-history│  → /history/{ticker}.parquet
│  - fetch-day    │  → /YYYY/MM/DD/{ticker}.parquet
│  - health       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Storage Account │
│  - /history/    │  (Dados históricos completos)
│  - /YYYY/MM/DD/ │  (Séries de 90 dias)
│  - /models/     │  (Modelos treinados)
│  - /hyperparams/│  (Configurações por ticker)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stock-Service  │  (Azure Function) [PENDENTE]
│  - /train       │  → Treina modelo LSTM
│  - /predict     │  → Retorna predição D+1
│  - /health      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Cosmos DB     │  [PENDENTE]
│  - model_versions│  (Versões de modelos)
│  - metrics      │  (Métricas de treinamento)
│  - predictions  │  (Cache de predições)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  API Management │  [PENDENTE]
│  - /data/*      │  (Endpoints do Data-Service)
│  - /stock/*     │  (Endpoints do Stock-Service)
└─────────────────┘
```

---

## ✅ Status do Projeto

### FASE 1: Data-Service (Azure Function) - ✅ **CONCLUÍDO**

**Status**: Implementado e funcional

- [x] Endpoint `fetch-history` (busca histórico completo + feature engineering)
- [x] Endpoint `fetch-day` (busca últimos 90 dias + feature engineering)
- [x] Health check `/health` implementado
- [x] Application Insights habilitado
- [x] Feature Engineering com 23 indicadores técnicos
- [x] Integração com Azure Storage Account
- [x] Suporte a múltiplos tickers configuráveis

**Outputs**:
- `/history/{ticker}.parquet` - Dados históricos completos com features
- `/YYYY/MM/DD/{ticker}.parquet` - Séries temporais de 90 dias para predição

**Endpoints Disponíveis**:
- `GET /api/fetch-history` - Busca histórico completo
- `GET /api/fetch-day` - Busca últimos 90 dias
- `GET /api/health` - Health check

---

### FASE 2: Modelo LSTM (Desenvolvimento Local) - ✅ **CONCLUÍDO**

**Status**: Modelo desenvolvido e testado localmente

#### 2.1 Arquitetura do Modelo ✅

- [x] Classe `StocksLSTM` usando PyTorch Lightning
- [x] Input: 23 features, Sequência: 70-90 dias (configurável por ticker)
- [x] Output: Adj Close (D+1)
- [x] Hiperparâmetros configuráveis por ticker (JSON)

**Arquitetura**:
```
Input (23 features × 70-90 timesteps) 
  → LSTM₁ (200 hidden) 
  → Dropout (0.2) 
  → LSTM₂ (100 hidden) 
  → Dropout (0.2) 
  → LSTM₃ (100 hidden, 2 layers) 
  → Dropout (0.2) 
  → Linear 
  → Output (Adj Close D+1)
```

#### 2.2 Pipeline de Treinamento ✅

- [x] Carregamento de dados de `/history/{ticker}.parquet`
- [x] Split 75/15/10 (treino/validação/teste)
- [x] Normalização com MinMaxScaler (0-1)
- [x] DataLoader com sequências temporais
- [x] Training loop com PyTorch Lightning
- [x] Early stopping e learning rate scheduler
- [x] Cálculo de métricas:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² (Coeficiente de Determinação)
  - Acurácia direcional

#### 2.3 Testes Locais ✅

- [x] Treinamento testado com PETR4, ITUB4, VALE3
- [x] Validação de predições
- [x] Salvamento de modelo + scaler localmente
- [x] Resultados documentados

**Resultados Obtidos (PETR4)**:
- R²: **63.26%**
- MAPE: **1.40%**
- MAE: R$ 0.54
- Acurácia Direcional: **72.22%**

**Notebooks Disponíveis**:
- `LSTM_BASE.ipynb` - Template base
- `LSTM_PETR4.ipynb` - Modelo otimizado para PETR4
- `LSTM_ITUB4.ipynb` - Modelo otimizado para ITUB4
- `LSTM_VALE3.ipynb` - Modelo otimizado para VALE3

---

### FASE 3: Infraestrutura Azure - ⏳ **PENDENTE**

#### 3.1 Cosmos DB

- [ ] Provisionar recurso Cosmos DB
- [ ] Configurar consistency level
- [ ] Criar container `model_versions`
- [ ] Criar container `training_metrics`
- [ ] Criar container `predictions`

#### 3.2 Storage Account

- [ ] Criar pasta `/hyperparameters` (JSONs de hiperparâmetros por ticker)
- [ ] Criar pasta `/models` (modelos treinados versionados)
- [ ] Estruturar organização de arquivos

#### 3.3 Application Insights

- [x] Habilitado no Data-Service
- [ ] Habilitar no Stock-Service (quando criado)
- [ ] Configurar retention period
- [ ] Configurar alertas básicos (opcional)

---

### FASE 4: Stock-Service (Azure Function) - ⏳ **PENDENTE**

#### 4.1 Setup Inicial

- [ ] Criar Function App `tc-4-stock-service`
- [ ] Configurar variáveis de ambiente
- [ ] Habilitar Application Insights
- [ ] Implementar health check `/health`

#### 4.2 Endpoint `/train`

- [ ] Carregar hiperparâmetros de `/hyperparameters/{ticker}.json`
- [ ] Carregar dados históricos de `/history/{ticker}.parquet`
- [ ] Executar pipeline de treinamento
- [ ] Gerenciar versionamento (Cosmos DB)
- [ ] Salvar modelo em `/models/{ticker}_v{version}.pt`
- [ ] Salvar scaler em `/models/{ticker}_v{version}_scaler.pkl`
- [ ] Salvar métricas em Cosmos DB

#### 4.3 Endpoint `/predict`

- [ ] Validar versão do modelo
- [ ] Implementar cache de predições (Cosmos DB)
- [ ] Carregar últimos 90 dias para predição
- [ ] Retornar predição com flag `from_cache`

#### 4.4 Deploy

- [ ] Criar Dockerfile (se necessário)
- [ ] Deploy via Azure CLI ou VS Code Extension
- [ ] Testar endpoints individualmente
- [ ] Validar integração com Storage e Cosmos

---

### FASE 5: API Management (APIM) - ⏳ **PENDENTE**

#### 5.1 Setup APIM

- [ ] Criar recurso API Management (Developer tier)
- [ ] Aguardar provisionamento
- [ ] Configurar domínio customizado (opcional)

#### 5.2 Importar APIs

- [ ] Importar Data-Service como API `/data`
- [ ] Importar Stock-Service como API `/stock`

#### 5.3 Configurar Segurança

- [ ] Mudar `authLevel` para `ANONYMOUS` nas Functions
- [ ] Criar Subscription Keys (principal + secundária)
- [ ] Configurar política de validação de header
- [ ] Testar autenticação

#### 5.4 Políticas e Configurações

- [ ] Configurar CORS
- [ ] Configurar rate limiting (throttling)
- [ ] Configurar cache de respostas (opcional)
- [ ] Habilitar logging detalhado

#### 5.5 Documentação

- [ ] Gerar documentação Swagger/OpenAPI automática
- [ ] Exportar especificação OpenAPI
- [ ] Disponibilizar developer portal (opcional)

---

### FASE 6: Dashboards e Monitoramento - ⏳ **PENDENTE**

#### 6.1 Dashboard de Infraestrutura (Application Insights)

- [ ] Uptime (disponibilidade dos serviços)
- [ ] Tempo médio de resposta por endpoint
- [ ] Taxa de sucesso (2xx vs 4xx/5xx)
- [ ] Throughput (requests/minuto)
- [ ] Latência P95/P99
- [ ] Erros e Exceptions

#### 6.2 Dashboard de Métricas de ML (Cosmos DB)

- [ ] Tabela de versões com métricas
- [ ] Gráfico de evolução do MAE por versão
- [ ] Comparação de métricas entre versões
- [ ] Acurácia direcional por versão e ticker
- [ ] Total de predições realizadas
- [ ] Última predição por ticker

#### 6.3 Montagem dos Dashboards

- [ ] Criar dashboard no Portal Azure
- [ ] Organizar layout (infra + ML)
- [ ] Compartilhar dashboard

#### 6.4 Alertas

- [ ] Alerta se uptime < 95%
- [ ] Alerta se tempo médio de resposta > 5s
- [ ] Alerta se taxa de erro > 5%
- [ ] Alerta se treinamento falhar

---

### FASE 7: Testes End-to-End - ⏳ **PENDENTE**

#### 7.1 Fluxo de Treinamento

- [ ] Chamar `/stock/train` via APIM
- [ ] Verificar modelo salvo
- [ ] Verificar registro em Cosmos DB
- [ ] Validar métricas

#### 7.2 Fluxo de Predição

- [ ] Chamar `/stock/predict` (primeira vez)
- [ ] Verificar cache funcionando
- [ ] Testar `forcePredict: true`

#### 7.3 Validação de Autenticação

- [ ] Testar sem Subscription Key (401)
- [ ] Testar com chave inválida (403)
- [ ] Testar com chave válida (200)
- [ ] Validar rate limiting (429)

#### 7.4 Validação de Monitoramento

- [ ] Verificar logs no Application Insights
- [ ] Verificar dashboards atualizando
- [ ] Testar health checks

---

### FASE 8: Documentação e Entrega - ⏳ **PENDENTE**

#### 8.1 Documentação Técnica

- [x] README.md principal (este arquivo)
- [ ] Diagrama de arquitetura visual
- [ ] Documentação de API (OpenAPI/Swagger)
- [ ] Exemplos de uso (curl/Postman)
- [ ] Troubleshooting comum

#### 8.2 Vídeo Demonstrativo

- [ ] Gravar screencast (6-8 minutos)
- [ ] Editar vídeo
- [ ] Adicionar narração
- [ ] Hospedar (YouTube/Vimeo)

#### 8.3 Organização do Repositório

- [x] Estrutura de pastas organizada
- [ ] Adicionar `.gitignore` completo
- [ ] Incluir exemplos de configuração (sem secrets)
- [ ] Adicionar LICENSE

---

## 📦 Pré-requisitos

### Local Development

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)
- Jupyter Lab (para notebooks)
- Azure CLI (para deploy)
- VS Code com extensão Azure Functions (opcional)

### Azure Resources

- Azure Subscription
- Storage Account (já configurado)
- Cosmos DB Account (pendente)
- Application Insights (já configurado no Data-Service)
- API Management (pendente)

---

## 📁 Estrutura do Projeto

```
tech-challenge-04/
├── data-service/              # Azure Function - Coleta de Dados
│   ├── function_app.py        # Endpoints: fetch-history, fetch-day, health
│   ├── host.json              # Configuração do Function App
│   ├── local.settings.json    # Variáveis de ambiente (local)
│   ├── requirements.txt       # Dependências Python
│   ├── pyproject.toml         # Configuração do projeto
│   └── utils/
│       ├── config.py          # Gerenciamento de configurações
│       ├── storage.py         # Cliente Azure Storage
│       ├── yfinance_client.py # Cliente yfinance
│       ├── parquet_handler.py # Manipulação de arquivos Parquet
│       └── feature_engineering.py  # 23 indicadores técnicos
│
├── stock-service/             # Azure Function - ML Service [PENDENTE]
│   ├── function_app.py        # Endpoints: train, predict, health
│   ├── model.py               # Classe LSTM (PyTorch Lightning)
│   └── ...
│
├── notebooks/                 # Desenvolvimento Local do Modelo
│   ├── LSTM_BASE.ipynb        # Template base
│   ├── LSTM_PETR4.ipynb       # Modelo otimizado PETR4
│   ├── LSTM_ITUB4.ipynb       # Modelo otimizado ITUB4
│   ├── LSTM_VALE3.ipynb       # Modelo otimizado VALE3
│   ├── configs/               # Hiperparâmetros por ticker
│   │   ├── PETR4.json
│   │   ├── ITUB4.json
│   │   └── VALE3.json
│   ├── data/
│   │   ├── train/             # Dados de treino (.parquet)
│   │   └── predict/           # Dados para predição
│   ├── models/                # Modelos salvos (.ckpt)
│   └── lightning_logs/        # Logs do TensorBoard
│
├── docs/                      # Documentação [PENDENTE]
│   ├── architecture_diagram.png
│   ├── openapi_spec.yaml
│   └── video_demo_link.md
│
├── dashboards/                # Dashboards Azure [PENDENTE]
│   └── ...
│
└── README.md                  # Este arquivo
```

---

## 🚀 Instalação e Setup

### 1. Clonar Repositório

```bash
git clone <repository-url>
cd tech-challenge-04
```

### 2. Setup Data-Service (Local)

```bash
cd data-service

# Instalar dependências
uv sync
# ou
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp local.settings.json.example local.settings.json
# Editar local.settings.json com suas credenciais Azure
```

**Variáveis de Ambiente Necessárias**:
- `AzureWebJobsStorage` - Connection string do Storage Account
- `TICKERS` - Lista de tickers (ex: "PETR4,ITUB4,VALE3")
- `CONTAINER_NAME` - Nome do container no Storage

### 3. Setup Notebooks (Local)

```bash
cd notebooks

# Instalar dependências
uv sync

# Ativar ambiente virtual
source .venv/bin/activate

# Iniciar Jupyter
jupyter lab
```

### 4. Executar Data-Service Localmente

```bash
cd data-service
func start
```

**Endpoints Locais**:
- `http://localhost:7071/api/fetch-history`
- `http://localhost:7071/api/fetch-day`
- `http://localhost:7071/api/health`

### 5. Deploy para Azure

```bash
# Login no Azure
az login

# Deploy do Data-Service
cd data-service
func azure functionapp publish <function-app-name>
```

---

## 📊 Features Utilizadas (23 Indicadores)

| Categoria | Indicadores |
|-----------|-------------|
| **Preço** | Open, High, Low, Adj Close |
| **Volume** | Volume, relative_volume, volume_ratio_5 |
| **Momentum** | RSI (7, 14), Stochastic K, ROC |
| **Tendência** | MA (3, 5, 9), distance_ma3, distance_ma9 |
| **Volatilidade** | volatility_5d, volatility_ratio, Bollinger Position |
| **MACD** | macd_histogram |
| **Outros** | gap, return_1d, return_3d |

---

## 📈 Resultados Obtidos

### PETR4 (Petrobras)

| Métrica | Valor |
|---------|-------|
| **R²** | 63.26% |
| **MAPE** | 1.40% |
| **MAE** | R$ 0.54 |
| **Acurácia Direcional** | 72.22% |

### ITUB4 (Itaú Unibanco)

🔧 Em otimização...

### VALE3 (Vale)

🔧 Em otimização...

---

## 🔧 Tecnologias Utilizadas

- **Python** 3.11+
- **PyTorch** 2.x
- **PyTorch Lightning** 2.x
- **Azure Functions** (Python)
- **Azure Storage Account** (Blob Storage)
- **Azure Cosmos DB** (pendente)
- **Azure API Management** (pendente)
- **Application Insights**
- **yfinance** (coleta de dados)
- **pandas** / **numpy** (manipulação de dados)
- **scikit-learn** (normalização, métricas)

---

## 📚 Documentação Detalhada

### Data-Service

O Data-Service é responsável por coletar dados históricos e diários das ações, aplicar feature engineering e salvar em formato Parquet no Azure Storage.

**Endpoints**:

1. **GET /api/fetch-history**
   - Busca histórico completo configurado
   - Aplica feature engineering (modo treino)
   - Salva em `/history/{ticker}.parquet`

2. **GET /api/fetch-day**
   - Busca últimos 90 dias
   - Aplica feature engineering (modo predição)
   - Salva em `/YYYY/MM/DD/{ticker}.parquet`

3. **GET /api/health**
   - Health check do serviço
   - Retorna status e timestamp

### Modelo LSTM

O modelo LSTM foi desenvolvido localmente usando PyTorch Lightning. Cada ticker possui sua própria configuração de hiperparâmetros em `notebooks/configs/{TICKER}.json`.

**Arquitetura**:
- 3 camadas LSTM empilhadas
- Dropout para regularização
- Normalização MinMax separada para features e target
- Early stopping e learning rate scheduler

**Métricas Calculadas**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coeficiente de Determinação)
- Acurácia Direcional

---

## 🗺️ Roadmap

### Próximos Passos Imediatos

1. **FASE 3**: Provisionar Cosmos DB e estruturar Storage Account
2. **FASE 4**: Desenvolver e deployar Stock-Service
3. **FASE 5**: Configurar API Management
4. **FASE 6**: Criar dashboards de monitoramento
5. **FASE 7**: Executar testes end-to-end
6. **FASE 8**: Finalizar documentação e vídeo

### Melhorias Futuras

- [ ] Implementar retreinamento automático periódico
- [ ] Adicionar mais indicadores técnicos
- [ ] Experimentar outras arquiteturas (Transformer, GRU)
- [ ] Implementar ensemble de modelos
- [ ] Adicionar predição de múltiplos dias (D+2, D+3, etc.)
- [ ] Dashboard web interativo (Streamlit/React)
- [ ] Notificações via email/Teams quando predições excedem threshold

---

## 🤝 Contribuindo

Este é um projeto acadêmico desenvolvido para o **Tech Challenge 04** da FIAP.

---

## 📄 Licença

Este projeto é de uso acadêmico.

---

## 📞 Contatos e Recursos

- **Repositório Git**: [(https://github.com/rebertmatheus/tech-challenge-04)]
- **Azure Portal**: [adicionar link do resource group]
- **Dashboard de Monitoramento**: [adicionar link quando criado]
- **Documentação API**: [adicionar link do APIM portal]

---

**Documento atualizado em**: Janeiro de 2025  
**Versão**: 1.0  
**Autor**: Equipe Tech Challenge 04

