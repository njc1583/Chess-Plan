import torch 
import pandas as pd 
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import dataset 
from torchvision import transforms
from PIL import Image

import DataUtils

class SplitImageDataset(Dataset):
    def __init__(self, metadata_csv, dataset_size=0, 
        use_depth=False, color_transform=None, depth_transform=None):
        
        self.metadata = pd.read_csv(metadata_csv)
        
        self.dataset_size = dataset_size

        self.use_depth = use_depth

        self.color_transform = color_transform
        self.depth_transform = depth_transform

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size 
        
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filenames = self.metadata.iloc[index, 0:2]

        piece = self.metadata.iloc[index,2]
        
        color_img = Image.fromarray(io.imread(filenames[0]), 'RGB')

        if self.color_transform:
            color_img = self.color_transform(color_img)

        if self.use_depth:
            depth_img = Image.fromarray(io.imread(filenames[1]), 'L')

            if self.depth_transform is not None:
                depth_img = self.depth_transform(depth_img)

        if self.use_depth:
            concat_img = torch.cat((color_img,depth_img), dim=0)
        else:
            concat_img = color_img

        return (concat_img, piece) 

# A simple test
if __name__ == '__main__':
    color_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    depth_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    print(f'Retrieving One Item')

    dset_full = SplitImageDataset('./image_dataset/metadata.csv',
        use_depth=True, 
        dataset_size=10,
        color_transform=color_transforms, 
        depth_transform=depth_transforms
    )

    im, c = dset_full[0] 

    print(im.shape, c)

    training_loader = DataLoader(dset_full, batch_size=5, shuffle=False, num_workers=0)

    for imgs,classes in training_loader:
        print(imgs.shape, classes.shape)