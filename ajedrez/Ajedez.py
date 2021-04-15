import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import math
import time
import random
from tqdm import tqdm
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader 

from DataProcessor import AjedrezDataset

NUM_CLASSES = 13

class Ajedrez(nn.Module):
    def __init__(self):
        super(Ajedrez, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 13, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        out = self.seq(x)

        # Output size should be [Batch Size x 13 x 64]
        out = out.reshape(B, NUM_CLASSES, 64)

        return out

class AJTrainer:
    def __init__(self, train_loader, test_loader):
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Ajedrez = Ajedrez().to(self._dev)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()

    def train(self, n_epochs=10):
        for epoch in tqdm(range(n_epochs)):
            e_loss = 0.0

            for imgs,labels in self.train_loader:
                imgs = imgs.to(self._dev)
                labels = labels.to(self._dev)

                preds = self.Ajedrez.forward(imgs)

                loss = self._get_loss(preds, labels)

                e_loss += loss.item()

                loss.backward()

            print(f'loss for epoch: {epoch}: {e_loss}')

    def _get_loss(self, pred, act):
        piece_bce = self.criterion(pred, act) 

        empty_pred = torch.zeros_like(pred) 
        empty_act = torch.zeros_like(act)  

        empty_pred[pred != 0] = 1
        empty_act[act != 0] = 1 

        empty_bce = self.criterion(empty_pred, empty_act)

        return (piece_bce + empty_bce).mean()  


if __name__ == "__main__":
    ts = transforms.Compose([transforms.ToTensor()])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts)

    loader = DataLoader(dset, batch_size=16, shuffle=False, num_workers=2)

    trainer = AJTrainer(loader, None)

    trainer.train()