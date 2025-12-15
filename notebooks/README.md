# 📈 LSTM Stock Price Predictor

Modelo LSTM (Long Short-Term Memory) para previsão de preços de ações usando **PyTorch Lightning**.

## 🎯 Objetivo

Prever o preço de fechamento ajustado (Adj Close) de ações da B3 utilizando indicadores técnicos e dados históricos.

## 🏗️ Arquitetura do Modelo

```
Input (23 features) → LSTM₁ (200) → Dropout → LSTM₂ (100) → Dropout → LSTM₃ (100, 2 layers) → Linear → Output
```

- **3 camadas LSTM** empilhadas com dropout para regularização
- **Normalização MinMax** separada para features e target
- **Sequências temporais** configuráveis por ticker
- **Early Stopping** e **Learning Rate Scheduler** para otimização

## 📊 Features Utilizadas

| Categoria | Indicadores |
|-----------|-------------|
| **Preço** | Open, High, Low, Adj Close |
| **Volume** | Volume, relative_volume, volume_ratio_5 |
| **Momentum** | RSI (7, 14), Stochastic K, ROC |
| **Tendência** | MA (3, 5, 9), distance_ma3, distance_ma9 |
| **Volatilidade** | volatility_5d, volatility_ratio, Bollinger Position |
| **MACD** | macd_histogram |
| **Outros** | gap, return_1d, return_3d |

## 📁 Estrutura do Projeto

```
notebooks/
├── LSTM+IAExpertAcademyV3.ipynb   # Notebook principal
├── configs/                       # Configurações por ticker
│   ├── PETR4.json
│   └── ITUB4.json
├── data/
│   ├── train/                     # Dados de treino (.parquet)
│   └── predict/                   # Dados para previsão
├── models/                        # Modelos salvos (.ckpt)
├── checkpoints/                   # Checkpoints durante treino
└── lightning_logs/                # Logs do TensorBoard
```

## ⚙️ Configuração por Ticker

Cada ação possui seu próprio arquivo JSON de configuração em `configs/`:

```json
{
  "TRAIN_RATIO": 0.75,
  "VAL_RATIO": 0.15,
  "DF_SIZE": 750,
  "LEARNING_RATE": 0.001,
  "WEIGHT_DECAY": 5e-6,
  "SEQUENCE_LENGTH": 70,
  "BATCH_SIZE": 16,
  "DROPOUT_VALUE": 0.2,
  "EPOCHS": 200,
  ...
}
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)

### Setup

```bash
# Clonar e entrar no diretório
cd notebooks

# Instalar dependências
uv sync

# Ativar ambiente virtual
source .venv/bin/activate
```

## 💻 Uso

### 1. Iniciar Jupyter

```bash
jupyter lab
```

### 2. Abrir o Notebook

Abra `LSTM+IAExpertAcademyV3.ipynb` e execute as células sequencialmente.

### 3. Trocar de Ticker

Edite o arquivo `configs/<TICKER>.json` ou crie um novo para outra ação.

```python
config = Config("PETR4")  # ou "ITUB4"
```

## 📈 Métricas de Avaliação

| Métrica | Descrição |
|---------|-----------|
| **MAE** | Erro Absoluto Médio (em R$) |
| **RMSE** | Raiz do Erro Quadrático Médio |
| **MAPE** | Erro Percentual Absoluto Médio |
| **R²** | Coeficiente de Determinação |
| **Acurácia Direcional** | % de acertos na direção do preço |

## 🏆 Resultados

### PETR4 (Petrobras)

| Métrica | Valor |
|---------|-------|
| R² | **63.26%** |
| MAPE | **1.40%** |
| MAE | R$ 0.54 |
| Acurácia Direcional | 72.22% |

### ITUB4 (Itaú Unibanco)

🔧 Em otimização...

## 🛠️ Tecnologias

- **PyTorch** 2.x
- **PyTorch Lightning** 2.x
- **scikit-learn** (MinMaxScaler, métricas)
- **pandas** / **numpy** (manipulação de dados)
- **matplotlib** (visualizações)

## 📝 Notas

- O modelo é otimizado **por ticker** — cada ação pode ter hiperparâmetros diferentes
- Previsão de **preços absolutos** pode ter dificuldades com tendências fortes de alta/baixa
- Recomenda-se **pelo menos 750 amostras** para treino adequado

## 👤 Autor

Desenvolvido como projeto acadêmico para a **FIAP** - IA Expert Academy.

## 📄 Licença

Este projeto é de uso acadêmico.
