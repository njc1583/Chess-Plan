import os
import torch 
import pandas as pd 
from skimage import io, transform
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils
from PIL import Image

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

        color_img = Image.fromarray(io.imread(filenames[0]), 'RGB')
        depth_img = Image.fromarray(io.imread(filenames[1]), 'L')

        pieces = self.metadata.iloc[index,2]

        classes = torch.tensor([int(x) for x in pieces.split(';')])

        if self.transform is not None:
            color_img = self.transform(color_img)
            depth_img = self.transform(depth_img)

        concat_img = torch.cat((color_img,depth_img), dim=0)

        return (concat_img, classes) 

# A simple test
if __name__ == '__main__':
    ts = transforms.Compose([
        transforms.ToTensor()
        ])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts)

    print(f'Fetching a single item')

    im, c = dset[0] 

    print(im.shape, c.shape)

    loader = DataLoader(dset, batch_size=16, shuffle=False, num_workers=2)

    i = 0

    print(f'Iterating through the dataset')

    for (i,(im,c)) in enumerate(loader):
        print(i, im.shape, c.shape)