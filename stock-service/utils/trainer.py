import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pandas as pd
import io
import joblib
import logging
import tempfile
import os

from .dataset import SequenceDataset
from .model import StocksLSTM
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Orquestra o pipeline completo de treinamento LSTM
    """
    
    def __init__(self):
        self.temp_dir = None
    
    def train(self, ticker: str, hyperparams: dict, df: pd.DataFrame):
        """
        Executa pipeline completo de treinamento
        
        Args:
            ticker: Ticker da ação
            hyperparams: Dicionário com hiperparâmetros
            df: DataFrame com dados históricos
        
        Returns:
            tuple: (model_bytes, scaler_bytes, metrics_dict)
                   model_bytes: Bytes do modelo checkpoint
                   scaler_bytes: Bytes dos scalers (joblib pickle)
                   metrics_dict: Dicionário com métricas de validação e teste
        """
        try:
            # Criar diretório temporário para checkpoints
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Diretório temporário criado: {self.temp_dir}")
            
            # 1. Preparar dados
            logger.info("Preparando dados...")
            df_clean = self._prepare_data(df, hyperparams)
            
            # 2. Criar datasets e dataloaders
            logger.info("Criando datasets e dataloaders...")
            train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = \
                self._create_datasets(df_clean, hyperparams)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=hyperparams["BATCH_SIZE"],
                shuffle=True,
                num_workers=0  # Azure Functions não suporta multiprocessing
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=hyperparams["BATCH_SIZE"],
                shuffle=False,
                num_workers=0
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=hyperparams["BATCH_SIZE"],
                shuffle=False,
                num_workers=0
            )
            
            # 3. Instanciar modelo
            logger.info("Instanciando modelo LSTM...")
            model = StocksLSTM(hyperparams)
            
            # 4. Configurar callbacks
            logger.info("Configurando callbacks...")
            callbacks = self._create_callbacks(hyperparams)
            
            # 5. Configurar trainer
            logger.info("Configurando PyTorch Lightning Trainer...")
            trainer = pl.Trainer(
                max_epochs=hyperparams["EPOCHS"],
                accelerator='cpu',  # Azure Functions Basic plan não tem GPU
                callbacks=callbacks,
                log_every_n_steps=hyperparams.get("LOG_EVERY_N_STEPS", 10),
                gradient_clip_val=hyperparams.get("GRADIENT_CLIP_VAL", 1.5),
                enable_progress_bar=False,  # Reduzir output em produção
                logger=False  # Desabilitar logger do Lightning
            )
            
            # 6. Executar treinamento
            logger.info("Iniciando treinamento...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # 7. Carregar melhor modelo do checkpoint
            logger.info("Carregando melhor modelo do checkpoint...")
            best_model_path = callbacks[1].best_model_path  # ModelCheckpoint é o segundo callback
            if best_model_path:
                model = StocksLSTM.load_from_checkpoint(best_model_path, config=hyperparams)
            else:
                logger.warning("Nenhum checkpoint encontrado, usando modelo atual")
            
            # 8. Gerar predições e calcular métricas
            logger.info("Gerando predições e calculando métricas...")
            metrics = self._calculate_all_metrics(model, val_loader, test_loader, 
                                                  val_dataset, test_dataset, target_scaler)
            
            # 9. Serializar modelo e scalers
            logger.info("Serializando modelo e scalers...")
            model_bytes = self._serialize_model(model, best_model_path)
            scaler_bytes = self._serialize_scalers(feature_scaler, target_scaler)
            
            return model_bytes, scaler_bytes, metrics
        
        except Exception as e:
            logger.exception(f"Erro durante treinamento para {ticker}")
            raise
        finally:
            # Limpar diretório temporário
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Diretório temporário removido: {self.temp_dir}")
    
    def _prepare_data(self, df: pd.DataFrame, hyperparams: dict) -> pd.DataFrame:
        """Prepara dados removendo colunas desnecessárias"""
        drop_columns = hyperparams.get("DROP_COLUMNS", [])
        df_clean = df.drop(columns=drop_columns, errors='ignore')
        return df_clean
    
    def _create_datasets(self, df: pd.DataFrame, hyperparams: dict):
        """
        Cria datasets de treino, validação e teste com split temporal
        """
        # Calcular índices de split (temporal, não aleatório!)
        sequence_length = hyperparams["SEQUENCE_LENGTH"]
        train_ratio = hyperparams.get("TRAIN_RATIO", 0.75)
        val_ratio = hyperparams.get("VAL_RATIO", 0.15)
        
        n_samples = len(df) - sequence_length
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        logger.info(f"Dataset original: {len(df)} registros")
        logger.info(f"Após criar sequências: {n_samples} samples")
        logger.info(f"Train: {train_end} samples ({train_ratio*100:.0f}%)")
        logger.info(f"Val: {val_end - train_end} samples ({val_ratio*100:.0f}%)")
        logger.info(f"Test: {n_samples - val_end} samples")
        
        # Criar dataset de TREINO
        train_dataset = SequenceDataset(
            df=df,
            sequence_length=sequence_length,
            feature_cols=hyperparams["FEATURE_COLS"],
            target_col=hyperparams["TARGET_COL"],
            feature_scaler=None,
            target_scaler=None,
            fit_scalers=True
        )
        
        # Salvar os scalers
        feature_scaler, target_scaler = train_dataset.get_scalers()
        
        # Criar datasets de VAL e TEST (usam os mesmos scalers)
        val_dataset = SequenceDataset(
            df=df,
            sequence_length=sequence_length,
            feature_cols=hyperparams["FEATURE_COLS"],
            target_col=hyperparams["TARGET_COL"],
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            fit_scalers=False
        )
        
        test_dataset = SequenceDataset(
            df=df,
            sequence_length=sequence_length,
            feature_cols=hyperparams["FEATURE_COLS"],
            target_col=hyperparams["TARGET_COL"],
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            fit_scalers=False
        )
        
        # Split temporal
        train_dataset.X = train_dataset.X[:train_end]
        train_dataset.y = train_dataset.y[:train_end]
        
        val_dataset.X = val_dataset.X[train_end:val_end]
        val_dataset.y = val_dataset.y[train_end:val_end]
        
        test_dataset.X = test_dataset.X[val_end:]
        test_dataset.y = test_dataset.y[val_end:]
        
        logger.info(f"Datasets criados: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler
    
    def _create_callbacks(self, hyperparams: dict):
        """Cria callbacks do PyTorch Lightning"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=hyperparams.get("ES_PATIENCE", 15),
            min_delta=hyperparams.get("ES_MIN_DELTA", 0.0001),
            mode='min',
            verbose=False
        )
        
        checkpoint = ModelCheckpoint(
            dirpath=self.temp_dir,
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=False
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        return [early_stopping, checkpoint, lr_monitor]
    
    def _calculate_all_metrics(self, model, val_loader, test_loader, 
                              val_dataset, test_dataset, target_scaler):
        """Calcula métricas de validação e teste"""
        # Gerar predições
        model.eval()
        with torch.no_grad():
            predictions_val = []
            targets_val = []
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs).flatten()
                predictions_val.extend(outputs.cpu().numpy())
                targets_val.extend(targets.cpu().numpy())
            
            predictions_test = []
            targets_test = []
            for batch in test_loader:
                inputs, targets = batch
                outputs = model(inputs).flatten()
                predictions_test.extend(outputs.cpu().numpy())
                targets_test.extend(targets.cpu().numpy())
        
        # Desnormalizar predições e targets
        predictions_val = target_scaler.inverse_transform(
            torch.tensor(predictions_val).reshape(-1, 1)
        ).flatten()
        targets_val = target_scaler.inverse_transform(
            torch.tensor(targets_val).reshape(-1, 1)
        ).flatten()
        
        predictions_test = target_scaler.inverse_transform(
            torch.tensor(predictions_test).reshape(-1, 1)
        ).flatten()
        targets_test = target_scaler.inverse_transform(
            torch.tensor(targets_test).reshape(-1, 1)
        ).flatten()
        
        # Calcular métricas
        metrics_val = calculate_metrics(targets_val, predictions_val, "Validação")
        metrics_test = calculate_metrics(targets_test, predictions_test, "Teste")
        
        return {
            "validation": metrics_val,
            "test": metrics_test
        }
    
    def _serialize_model(self, model, checkpoint_path: str = None) -> bytes:
        """Serializa modelo para bytes"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Ler arquivo checkpoint diretamente
            with open(checkpoint_path, 'rb') as f:
                return f.read()
        else:
            # Salvar modelo em memória
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            return buffer.read()
    
    def _serialize_scalers(self, feature_scaler, target_scaler) -> bytes:
        """Serializa scalers para bytes usando joblib"""
        buffer = io.BytesIO()
        joblib.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }, buffer)
        buffer.seek(0)
        return buffer.read()
