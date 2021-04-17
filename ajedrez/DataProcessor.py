import os
import torch 
import pandas as pd 
from skimage import io, transform
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils
from PIL import Image

IMAGE_SIZE = 640

class AjedrezDataset(Dataset):
    def __init__(self, metadata_csv, root_dir, transform=None, num_images=-1):
        self.metadata = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.transform = transform 
        self.num_images = num_images

    def __len__(self):
        if self.num_images != -1:
            return self.num_images 
        
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filenames = self.metadata.iloc[index, 0:2]

        color_img = Image.fromarray(io.imread(filenames[0]), 'RGB')
        depth_img = Image.fromarray(io.imread(filenames[1]), 'L')

        pieces = self.metadata.iloc[index,2]

        classes = torch.tensor([int(x) for x in pieces.split(';')], dtype=torch.uint8)

        if self.transform is not None:
            color_img = self.transform(color_img)
            depth_img = self.transform(depth_img)

        concat_img = torch.cat((color_img,depth_img), dim=0)

        B, H, W, C = 36, IMAGE_SIZE//10, IMAGE_SIZE//10, 4
        split_img = torch.zeros(B, C, 3*H, 3*W)

        classes_idx = torch.zeros((B,), dtype=torch.int64)

        # Retrieve non-empty classes
        piece_classes_idx = (classes != 0).nonzero().flatten()

        randomized_empty_idx = torch.randperm(31)[:4]
        empty_classes_idx = ((classes == 0).nonzero().flatten())[randomized_empty_idx]

        classes_idx[:32] = piece_classes_idx
        classes_idx[32:] = empty_classes_idx

        # print(classes)
        # print(classes_idx)
        # print(torch.gather(classes, 0, classes_idx))

        for split_img_idx,idx in enumerate(classes_idx):
            i = idx.item()

            row = i // 8
            col = i % 8

            split_img[split_img_idx] = concat_img[:,row*W:(row+3)*W,col*H:(col+3)*H]

        # for bx in range(8):
        #     for by in range(8):
        #         split_img[bx*8+by] = concat_img[:,bx*W:bx*W+3*W,by*H:by*H+3*H]

        out_classes = torch.gather(classes, 0, classes_idx).type(torch.long)

        return (split_img, out_classes) 

# A simple test
if __name__ == '__main__':
    ts = transforms.Compose([
        transforms.ToTensor()
        ])

    dset = AjedrezDataset('./image_dataset/metadata.csv', './.', ts)

    print(f'Fetching a single item')

    im, c = dset[0] 

    print(im.shape, c.shape)

    # print(c)

    # loader = DataLoader(dset, batch_size=16, shuffle=False, num_workers=2)

    # i = 0

    # print(f'Iterating through the dataset')

    # for (i,(im,c)) in enumerate(loader):
    #     print(i, im.shape, c.shape)