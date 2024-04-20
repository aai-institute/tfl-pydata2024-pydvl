import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class Resnet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return torch.nn.functional.softmax(self.model(x), dim=1)


class LitResnet18SmallBinary(pl.LightningModule):
    def __init__(self):
        super(LitResnet18SmallBinary, self).__init__()
        self.model = Resnet18Binary()

        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.precision = torchmetrics.classification.BinaryPrecision()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(preds, y))
        self.log("train_prec", self.precision(preds, y))
        self.log("train_recall", self.recall(preds, y))
        self.log("train_f1", self.f1(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(preds, y))
        self.log("val_prec", self.precision(preds, y))
        self.log("val_recall", self.recall(preds, y))
        self.log("val_f1", self.f1(preds, y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return optimizer
