import torch 
import pandas as pd 
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import dataset 
from torchvision import transforms
from PIL import Image

import DataUtils

class WholeImageDataset(Dataset):
    def __init__(self, metadata_csv, color_transform, dataset_size=0):
    
        print('Note: the WholeImageDataset is to be used only for visual debugging, and does not return any classes when indexing. Be cautious when using.')

        self.metadata = pd.read_csv(metadata_csv)
        
        self.dataset_size = dataset_size

        self.color_transform = color_transform

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size 
        
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filenames = 'image_dataset/rectified/%06d.png' % index
        
        color_img = io.imread(filenames)

        concat_img = DataUtils.split_image_pytorch(color_img, self.color_transform)

        return (torch.tensor(color_img), concat_img) 

# A simple test
if __name__ == '__main__':
    color_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f'Retrieving One Item')

    dset_full = WholeImageDataset('./image_dataset/metadata.csv',
        color_transform=color_transforms,
        dataset_size=10
    )

    color_img, concat_img = dset_full[0] 

    print(color_img.shape, concat_img.shape)

    training_loader = DataLoader(dset_full, batch_size=5, shuffle=False, num_workers=0)

    for color_img,concat_img in training_loader:
        print(color_img.shape, concat_img.shape)