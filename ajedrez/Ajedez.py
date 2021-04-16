import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import math
import time
import random

import torch.optim
from tqdm import tqdm

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split 

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

def get_loss(criterion, pred, act):
    piece_bce = criterion(pred, act) 

    # return piece_bce.mean()

    empty_pred = torch.zeros_like(pred) 
    empty_act = torch.zeros_like(act)  

    amax = pred.argmax(2, keepdim=True)

    empty_pred = torch.zeros_like(pred).scatter(2, amax, 1.0)
    empty_act[act != 0] = 1 

    empty_bce = criterion(empty_pred, empty_act)

    return (piece_bce + empty_bce).mean()  


def train(AJ, train_loader, test_loader, optimizer, lr_scheduler, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        for imgs,labels in tqdm(train_loader):
            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = AJ.forward(imgs)

            loss = get_loss(criterion, preds, labels)

            loss.backward()
            optimizer.step()

        AJ.eval()

        t_loss = 0.0

        for timg,tlabels in tqdm(test_loader):
            timg = timg.to(device)
            tlabels = tlabels.to(device)

            tpreds = AJ.forward(timg) 

            loss = get_loss(criterion, tpreds, tlabels)

            t_loss += loss.item() 

        t_loss /= len(test_loader)

        print(f'Test loss at epoch {epoch}: {t_loss}')

        lr_scheduler.step()

if __name__ == "__main__":
    ts = transforms.Compose([transforms.ToTensor()])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts)

    train_data, test_data = random_split(dset, [10_000, 1_000])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AJ = Ajedrez().to(device)

    sgd = optim.SGD(AJ.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=0.1)

    train(AJ, train_loader, test_loader, sgd, scheduler, device, 10)