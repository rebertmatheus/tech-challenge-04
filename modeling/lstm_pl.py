from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
import pytorch_lightning as pl


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)

        y_arr = torch.tensor(y, dtype=torch.float32)
        self.y = y_arr.squeeze(-1) if y_arr.ndim > 1 else y_arr

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size1: int = 64, hidden_size2: int = 32, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size2, 1)

        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Log the training loss for tracking.
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Use prog_bar=True so that Lightning prints val_loss in the progress bar
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            },
        }