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
        feature_cols = self.get_config("FEATURE_COLS")
        init_hidden_size = self.get_config("INIT_HIDDEN_SIZE")
        second_hidden_size = self.get_config("SECOND_HIDDEN_SIZE")
        dropout_value = self.get_config("DROPOUT_VALUE")
        num_layers = self.get_config("NUM_LAYERS")
        
        self.lstm1 = nn.LSTM(
            input_size=len(feature_cols), 
            hidden_size=init_hidden_size
        )
        self.lstm2 = nn.LSTM(
            init_hidden_size, 
            second_hidden_size
        )
        self.lstm3 = nn.LSTM(
            second_hidden_size, 
            second_hidden_size, 
            dropout=dropout_value, 
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(p=dropout_value)
        
        # Verificar se deve usar camadas FC adicionais (nn.Sequential)
        use_fc_layers = self.get_config("USE_FC_LAYERS", False)
        
        if use_fc_layers:
            # Usar nn.Sequential com múltiplas camadas FC
            fc_hidden_1 = self.get_config("FC_HIDDEN_1", 64)
            fc_hidden_2 = self.get_config("FC_HIDDEN_2", 32)
            fc_dropout = self.get_config("FC_DROPOUT", 0.1)
            
            self.fc = nn.Sequential(
                nn.Linear(second_hidden_size, fc_hidden_1),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc_hidden_1, fc_hidden_2),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc_hidden_2, 1)
            )
        else:
            # Manter original (1 camada)
            self.fc = nn.Linear(
                in_features=second_hidden_size,
                out_features=1
            )
        
        # Salvar hiperparâmetros para PyTorch Lightning
        self.save_hyperparameters()
    
    # Helper para obter valor do config (dict ou objeto)
    def get_config(self, key, default=None):
        return self.config.get(key, default) if isinstance(self.config, dict) else getattr(self.config, key, default)
    
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
        x = self.fc(x)
        
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
            lr=self.get_config("LEARNING_RATE"),
            weight_decay=self.get_config("WEIGHT_DECAY")
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.get_config("RLR_FACTOR"),
            patience=self.get_config("RLR_PATIENCE")
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
