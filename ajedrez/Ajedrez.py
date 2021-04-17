import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import math
import time
import random

import torch.optim
from tqdm.notebook import tqdm

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split 

from DataProcessor import AjedrezDataset

NUM_CLASSES = 13

class Ajedrez(nn.Module):
    def __init__(self, input_channels):
        super(Ajedrez, self).__init__()

        # self.seq = nn.Sequential(
        #     nn.Conv2d(4, 4, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(4, 8, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(8, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 13, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        self.input_channels = input_channels

        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, 4, 3, 1, 1),
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
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 832),
            nn.ReLU(),
            nn.Linear(832, 13),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.seq(x)
        
        # B, C, H, W = x.shape

        # Output size should be [Batch Size x 13 x 64]
        # out = out.reshape(B, NUM_CLASSES, 64)

        return out

def get_loss(criterion, pred, act):
    # BCE on which piece was selected
    piece_bce = criterion(pred, act) 

    return piece_bce

    # Correclty identifies if the tile was empty
    # 0 = empty, 1 = piece
    # empty_act = torch.zeros_like(act)  
    # empty_act[act != 0] = 1 

    # amax = pred.argmax(dim=1)    
    # empty_pred = torch.ones_like(act)
    # empty_pred[amax == 0] = 0 

    # print(empty_pred.shape, empty_act.shape)

    # bceloss = nn.BCELoss() 
    
    # empty_bce = bceloss(empty_pred, empty_act)

    # return (piece_bce + empty_bce).mean()  


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

            train_loss += loss.mean().item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        print(f'Training Loss at Epoch: {epoch}: {train_loss}')

        AJ.eval()

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

            test_loss += loss.mean().item() 

        test_loss /= len(test_loader)

        print(f'Test Loss at Epoch: {epoch}: {test_loss}')

        lr_scheduler.step()

if __name__ == "__main__":
    ts = transforms.Compose([transforms.ToTensor()])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts, 10)

    train_data, test_data = random_split(dset, [8, 2])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AJ = Ajedrez().to(device)

    sgd = optim.SGD(AJ.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=0.1)

    train(AJ, train_loader, test_loader, sgd, scheduler, device, 10)