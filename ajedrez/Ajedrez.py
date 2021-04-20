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

import DataUtils

from DataProcessor import AjedrezDataset

import torchvision.models
from torchvision.models import resnet

NUM_CLASSES = 13

class Ajedrez(nn.Module):
    def __init__(self, input_channels):
        super(Ajedrez, self).__init__()

        self.input_channels = input_channels

        self.pre_model = torchvision.models.resnet152(pretrained=True)

        for param in self.pre_model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.pre_model.fc.in_features
        self.pre_model.fc = nn.Linear(self.num_ftrs, 13)

    def forward(self, x):
        out = self.pre_model(x)
        
        return out

def get_loss(criterion, pred, act):
    # BCE on which piece was selected
    piece_bce = criterion(pred, act) 

    return piece_bce

def get_aj_loss(criterions, alphas, pred, act):
    assert len(criterions) == len(alphas) == len(pred) == 3

    nonempty_class = act != 0

    nonempty_actual = torch.zeros_like(act)
    nonempty_actual[nonempty_class] = 1

    white_class = (act >= 1) & (act <= 6)

    color_act = torch.zeros_like(act)
    color_act[white_class] = 1 
    color_act = color_act[nonempty_class]

    color_pred = pred[2][nonempty_class]

    class_loss = alphas[0] * criterions[0](pred[0], act)
    nonempty_loss = alphas[1] * criterions[1](pred[1], nonempty_actual)
    color_loss = alphas[2] * criterions[2](color_pred, color_act)

    return class_loss + nonempty_loss + color_loss


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

    dset = AjedrezDataset('./image_dataset/metadata.csv',
        dataset_size=10,
        use_depth=False,
        color_transform=ts, depth_transform=ts,
        full_image=False, 
        class_distribution=DataUtils.UNIFORM_DISTRIBUTION
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AJ = Ajedrez(3).to(device)

    data, classes = dset[0]
    data, classes = data.to(device), classes.to(device)

    out = AJ.forward(data)

    criterion = nn.CrossEntropyLoss()

    loss = get_loss(criterion, out, classes)

    print(loss.item())

    # (class_out, empty_out, color_out) = out = AJ.forward(data)

    # print(class_out.shape)
    # print(empty_out.shape)
    # print(color_out.shape)

    # class_criterion = nn.CrossEntropyLoss()
    # empty_criterion = nn.CrossEntropyLoss()
    # color_criterion = nn.CrossEntropyLoss()

    # criterions = (class_criterion, empty_criterion, color_criterion)
    # alphas = (0.8, 0.1, 0.1)

    # loss = get_aj_loss(criterions, alphas, out, classes)

    # print(loss.cpu().item())    