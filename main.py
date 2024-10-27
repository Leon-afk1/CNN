import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
from IPython.display import display
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import wandb
from pytorch_lightning.loggers import WandbLogger


@dataclass
class Config:
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "logs/"
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int = 1

config = Config()

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 128)  
        self.l2 = nn.Linear(128, 64)     
        self.l3 = nn.Linear(64, 10)        

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        
        x = self.l3(x)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

train_transforms = transforms.Compose([
    transforms.RandomCrop(28, padding=4),       
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

train_ds = MNIST(config.data_dir, train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

val_transforms = transforms.ToTensor()
val_ds = MNIST(config.data_dir, train=False, download=True, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=config.batch_size)

test_ds = MNIST(config.data_dir, train=False, download=True, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=config.batch_size)

wandb.init(project="MNIST-Classification")
wandb_logger = WandbLogger(project="MNIST-Classification")

mnist_model = MNISTModel()

trainer = pl.Trainer(
    accelerator=config.accelerator,
    devices=config.devices,
    max_epochs=config.max_epochs,
    logger=wandb_logger  
)

trainer.fit(mnist_model, train_loader, val_loader)

trainer.test(dataloaders=test_loader)

wandb.finish()


