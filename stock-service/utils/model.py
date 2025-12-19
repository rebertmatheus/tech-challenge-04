import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)

class StocksLSTM(pl.LightningModule):
    """
    Modelo LSTM para predição de preços de ações
    Arquitetura: 3 camadas LSTM com dropout
    """
    
    def __init__(self, config):
        """
        Args:
            config: Objeto de configuração com hiperparâmetros
                   Deve ter: FEATURE_COLS, INIT_HIDDEN_SIZE, SECOND_HIDDEN_SIZE,
                            NUM_LAYERS, DROPOUT_VALUE, LEARNING_RATE, WEIGHT_DECAY,
                            RLR_FACTOR, RLR_PATIENCE
        """
        super().__init__()
        self.config = config
        
        # Camadas LSTM
        self.lstm1 = nn.LSTM(
            input_size=len(config["FEATURE_COLS"]), 
            hidden_size=config["INIT_HIDDEN_SIZE"]
        )
        self.lstm2 = nn.LSTM(
            config["INIT_HIDDEN_SIZE"], 
            config["SECOND_HIDDEN_SIZE"]
        )
        self.lstm3 = nn.LSTM(
            config["SECOND_HIDDEN_SIZE"], 
            config["SECOND_HIDDEN_SIZE"], 
            dropout=config["DROPOUT_VALUE"], 
            num_layers=config["NUM_LAYERS"]
        )
        self.dropout = nn.Dropout(p=config["DROPOUT_VALUE"])
        self.linear = nn.Linear(
            in_features=config["SECOND_HIDDEN_SIZE"], 
            out_features=1
        )
        
        # Salvar hiperparâmetros para PyTorch Lightning
        self.save_hyperparameters()
    
    def forward(self, x):
        """
        Forward pass do modelo
        
        Args:
            x: Tensor de input (batch_size, sequence_length, num_features)
        
        Returns:
            Tensor de output (batch_size, 1)
        """
        # Permutar para (sequence_length, batch_size, num_features)
        x = x.permute(1, 0, 2)
        
        # Passar pelas camadas LSTM
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        
        # Pegar última saída da sequência
        x = x[-1]
        x = self.dropout(x)
        x = self.linear(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        """Step de treinamento"""
        inputs, targets = batch
        outputs = self(inputs).flatten()
        loss = F.mse_loss(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Step de validação"""
        inputs, targets = batch
        outputs = self(inputs).flatten()
        loss = F.mse_loss(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Step de teste"""
        inputs, targets = batch
        outputs = self(inputs).flatten()
        loss = F.mse_loss(outputs, targets)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Step de predição"""
        inputs, _ = batch
        return self(inputs).flatten()
    
    def configure_optimizers(self):
        """Configura otimizador e scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config["WEIGHT_DECAY"]
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config["RLR_FACTOR"],
            patience=self.config["RLR_PATIENCE"]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
