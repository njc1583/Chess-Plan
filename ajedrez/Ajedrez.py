import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import math
import time
import random

import torch.optim
from torchvision.transforms.transforms import Normalize
from tqdm.notebook import tqdm

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split 

from DataProcessor import AjedrezDataset

from torchvision.models import resnet

NUM_CLASSES = 13

class Ajedrez(nn.Module):
    def __init__(self, input_channels):
        super(Ajedrez, self).__init__()

        self.input_channels = input_channels

        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            resnet.BasicBlock(8, 8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            resnet.BasicBlock(16, 16),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 13),
            nn.ReLU()
        )

        self._weight_init()

    def forward(self, x):
        out = self.seq(x)
        
        return out

    def _weight_init(self):
        for child in self.seq.children():
            if isinstance(child, nn.Conv2d):
                for name, param in child.named_parameters():
                    if name in ['bias']:
                        nn.init.zeros_(param)
                    else:
                        nn.init.xavier_normal_(param, nn.init.calculate_gain('relu'))

def get_loss(criterion, pred, act):
    # BCE on which piece was selected
    piece_bce = criterion(pred, act) 

    return piece_bce

def train(AJ, train_loader, test_loader, optimizer, lr_scheduler, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0

        AJ.train()

        for train_images,train_labels in tqdm(train_loader):
            optimizer.zero_grad()

            if len(train_images.shape) == 5:
                B1, B2, C, H, W = train_images.shape

                train_images = train_images.reshape(B1*B2, C, H, W)
                train_labels = train_labels.reshape(B1*B2,)

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            preds = AJ.forward(train_images)

            loss = get_loss(criterion, preds, train_labels)

            # print(f'Loss during epoch: {loss}')

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        print(f'Training Loss at Epoch: {epoch}: {train_loss}')

        AJ.eval()
        with torch.no_grad():
            test_loss = 0.0

            for test_images,test_labels in tqdm(test_loader):
                if len(test_images.shape) == 5:
                    B1, B2, C, H, W = test_images.shape
                
                    test_images = test_images.reshape(B1*B2,C,H,W)
                    test_labels = test_labels.reshape(B1*B2,)

                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                tpreds = AJ.forward(test_images) 

                loss = get_loss(criterion, tpreds, test_labels)

                test_loss += loss.item() 

            test_loss /= len(test_loader)

            print(f'Test Loss at Epoch: {epoch}: {test_loss}')

        lr_scheduler.step()

if __name__ == "__main__":
    ts = transforms.Compose([
        transforms.ToTensor()])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts, 10)

    train_data, test_data = random_split(dset, [8, 2])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AJ = Ajedrez().to(device)

    sgd = optim.SGD(AJ.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=0.1)

    train(AJ, train_loader, test_loader, sgd, scheduler, device, 10)