import os
import torch 
import pandas as pd 
from skimage import io, transform
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils

class AjedrezDataset(Dataset):
    def __init__(self, metadata_csv, root_dir, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.transform = transform 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filenames = self.metadata.iloc[index, 0:2]

        color_img = torch.from_numpy(io.imread(filenames[0]).astype(np.float32))
        depth_img = torch.from_numpy(io.imread(filenames[1]).astype(np.float32))

        h, w = depth_img.shape

        depth_img = depth_img.reshape(h, w, 1)

        pieces = self.metadata.iloc[index,2]

        concat_img = torch.cat((color_img,depth_img), 2)

        classes = torch.tensor([int(x) for x in pieces.split(';')])

        return {'image': concat_img, 'classes': classes}

# A simple test
if __name__ == '__main__':
    dset = AjedrezDataset('./image_dataset/metadata.csv', './.')

    a = dset[0]