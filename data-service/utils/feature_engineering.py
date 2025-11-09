import logging
from typing import Optional
import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    Cria indicadores técnicos otimizados para predição de 1 dia.
    Foco em indicadores de curto prazo e alta sensibilidade.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_features(
        self, 
        df: Optional[pd.DataFrame] = None, 
        is_training_data: bool = True,
        target_days: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Cria os 15 indicadores técnicos essenciais para predição de curto prazo.

        Args:
            df: DataFrame com colunas obrigatórias: ['Close', 'Volume']
                Opcionais: ['Open', 'High', 'Low'] para indicadores OHLC
            is_training_data: Se True, calcula target. Se False, modo predição.
            target_days: Horizonte de predição (padrão: 1 dia)

        Returns:
            DataFrame com indicadores calculados ou None se df for None

        Features criadas (15 essenciais):
            1. rsi_7: RSI de 7 períodos (curto prazo)
            2. rsi_14: RSI de 14 períodos (médio prazo)
            3. macd_histogram: Histograma MACD
            4. ma3, ma5, ma9: Médias móveis curtas
            5. distance_ma3, distance_ma9: Distância das MAs (%)
            6. return_1d, return_3d: Retornos de 1 e 3 dias
            7. roc_3: Rate of Change de 3 dias
            8. volatility_5d, volatility_ratio: Volatilidade recente
            9. relative_volume, volume_ratio_5: Indicadores de volume
            10. bb_position: Posição nas Bollinger Bands
            11. stoch_k: Stochastic %K (se OHLC disponível)
            12. gap: Gap de abertura (se Open disponível)
        """

        # Validação inicial
        if df is None:
            self.logger.warning("DataFrame é None, não é possível criar indicadores.")
            return None

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

        # Validar colunas obrigatórias
        required_cols = ['Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias não encontradas: {missing_cols}")

        try:
            data = df.copy()

            # Tratar MultiIndex se necessário
            if isinstance(data.columns, pd.MultiIndex):
                ticker = data.columns.get_level_values(1)[0]
                data = data.xs(ticker, axis=1, level=1)
                self.logger.info(f"MultiIndex detectado, usando ticker: {ticker}")

            # Verificar disponibilidade de OHLC
            has_ohlc = all(col in data.columns for col in ['Open', 'High', 'Low', 'Close'])
            has_open = 'Open' in data.columns

            price_col = 'Close'

            self.logger.info(f"Iniciando cálculo de features. Shape inicial: {data.shape}")
            self.logger.info(f"OHLC disponível: {has_ohlc}, Open disponível: {has_open}")

            # ================================================
            # 1. RSI (Relative Strength Index)
            # ================================================

            # RSI 7 períodos (curto prazo - mais sensível)
            delta = data[price_col].diff().astype(float)
            gain_7 = (delta.where(delta > 0, 0)).ewm(alpha=1/7, adjust=False).mean()
            loss_7 = (-delta.where(delta < 0, 0)).ewm(alpha=1/7, adjust=False).mean()
            rs_7 = gain_7 / loss_7.replace(0, np.nan)
            data['rsi_7'] = 100 - (100 / (1 + rs_7))
            data['rsi_7'] = data['rsi_7'].fillna(50)  # Valor neutro

            # RSI 14 períodos (médio prazo - padrão)
            gain_14 = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss_14 = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs_14 = gain_14 / loss_14.replace(0, np.nan)
            data['rsi_14'] = 100 - (100 / (1 + rs_14))
            data['rsi_14'] = data['rsi_14'].fillna(50)

            # ================================================
            # 2. MACD (Moving Average Convergence Divergence)
            # ================================================

            ema12 = data[price_col].ewm(span=12, adjust=False).mean()
            ema26 = data[price_col].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            data['macd_histogram'] = macd - macd_signal

            # ================================================
            # 3. Moving Averages (Médias Móveis)
            # ================================================

            data['ma3'] = data[price_col].rolling(window=3).mean()
            data['ma5'] = data[price_col].rolling(window=5).mean()
            data['ma9'] = data[price_col].rolling(window=9).mean()

            # Distância das médias móveis (normalizada em %)
            data['distance_ma3'] = np.where(
                data['ma3'] != 0,
                (data[price_col] - data['ma3']) / data['ma3'] * 100,
                0
            )
            data['distance_ma9'] = np.where(
                data['ma9'] != 0,
                (data[price_col] - data['ma9']) / data['ma9'] * 100,
                0
            )

            # ================================================
            # 4. Returns (Retornos)
            # ================================================

            data['return_1d'] = data[price_col].pct_change(periods=1) * 100
            data['return_3d'] = data[price_col].pct_change(periods=3) * 100

            # ================================================
            # 5. ROC (Rate of Change)
            # ================================================

            data['roc_3'] = np.where(
                data[price_col].shift(3) != 0,
                ((data[price_col] - data[price_col].shift(3)) / 
                 data[price_col].shift(3)) * 100,
                0
            )

            # ================================================
            # 6. Volatilidade
            # ================================================

            returns_pct = data[price_col].pct_change()
            data['volatility_5d'] = returns_pct.rolling(window=5).std() * 100
            volatility_10d = returns_pct.rolling(window=10).std() * 100

            data['volatility_ratio'] = np.where(
                volatility_10d != 0,
                data['volatility_5d'] / volatility_10d,
                1.0
            )

            # ================================================
            # 7. Volume
            # ================================================

            vol_ma_5 = data['Volume'].rolling(window=5).mean()
            vol_ma_20 = data['Volume'].rolling(window=20).mean()

            data['relative_volume'] = np.where(
                vol_ma_20 != 0,
                data['Volume'] / vol_ma_20,
                1.0
            )

            data['volume_ratio_5'] = np.where(
                vol_ma_5 != 0,
                data['Volume'] / vol_ma_5,
                1.0
            )

            # ================================================
            # 8. Bollinger Bands
            # ================================================

            bb_ma = data[price_col].rolling(window=20).mean()
            bb_std = data[price_col].rolling(window=20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)

            bb_range = bb_upper - bb_lower
            data['bb_position'] = np.where(
                bb_range != 0,
                (data[price_col] - bb_lower) / bb_range,
                0.5  # Valor neutro (meio das bandas)
            )

            # ================================================
            # 9. Stochastic (apenas se OHLC disponível)
            # ================================================

            if has_ohlc:
                low_5 = data['Low'].rolling(window=5).min()
                high_5 = data['High'].rolling(window=5).max()

                data['stoch_k'] = np.where(
                    (high_5 - low_5) != 0,
                    ((data['Close'] - low_5) / (high_5 - low_5)) * 100,
                    50  # Valor neutro apenas quando range = 0
                )
                self.logger.info("Stochastic %K calculado (OHLC disponível)")
            else:
                self.logger.warning("Stochastic %K não calculado - OHLC não disponível")

            # ================================================
            # 10. Gap (apenas se Open disponível)
            # ================================================

            if has_open:
                data['gap'] = np.where(
                    data['Close'].shift(1) != 0,
                    (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100,
                    0
                )
                self.logger.info("Gap calculado (Open disponível)")
            else:
                data['gap'] = 0.0  # Sem gap se não tiver Open
                self.logger.info("Gap definido como 0 (Open não disponível)")

            # ================================================
            # TARGET (apenas para training)
            # ================================================

            if is_training_data:
                data['target'] = np.where(
                    data[price_col] != 0,
                    (data[price_col].shift(-target_days) / data[price_col] - 1) * 100,
                    0
                )

                # Remove linhas com NaN
                initial_rows = len(data)
                data = data.dropna()
                removed = initial_rows - len(data)

                self.logger.info(f"Target calculado: {target_days} dia(s) à frente")
                self.logger.info(f"Linhas removidas (NaN): {removed}")
                self.logger.info(f"Shape final (training): {data.shape}")

            else:
                # Modo predição: preenche NaN
                data = data.ffill().bfill()

                self.logger.info("Modo PREDIÇÃO: NaN preenchidos")
                self.logger.info(f"Shape final (predição): {data.shape}")

                if len(data) > 0:
                    last_date = data.index[-1] if hasattr(data.index, '__getitem__') else 'N/A'
                    self.logger.info(f"Última data: {last_date}")

            # Garantir que retorna DataFrame (não Series)
            if isinstance(data, pd.Series):
                self.logger.warning("Convertendo Series para DataFrame")
                data = data.to_frame()

            return data

        except Exception as e:
            self.logger.exception(f"Erro ao criar indicadores: {e}")
            raise