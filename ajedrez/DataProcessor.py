import torch 
import pandas as pd 
from skimage import io
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image

import DataUtils

class AjedrezDataset(Dataset):
    def __init__(self, metadata_csv, dataset_size=0, 
        use_depth=False, color_transform=None, depth_transform=None,
        full_image=False, class_distribution=None):
        
        self.metadata = pd.read_csv(metadata_csv)
        
        self.dataset_size = dataset_size

        self.use_depth = use_depth

        self.color_transform = color_transform
        self.depth_transform = depth_transform

        self.full_image = full_image
        self.class_distribution = class_distribution

        if not self.full_image and self.class_distribution is None:
            raise Exception('If not using full images; user must specify a class distribution')
        elif self.class_distribution is not None and len(self.class_distribution) != 13:
            raise Exception('Class distribution must be size 13')

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size 
        
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filenames = self.metadata.iloc[index, 0:2]

        pieces = self.metadata.iloc[index,2]
        classes = torch.tensor([int(x) for x in pieces.split(';')], dtype=torch.uint8)

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

        if self.full_image:
            B = 64
        else:
            B = sum(self.class_distribution)

        H = DataUtils.IMAGE_SIZE // 10
        W = DataUtils.IMAGE_SIZE // 10
        C = 4 if self.use_depth else 3

        split_img = torch.zeros(B, C, 2*H, W)

        if self.full_image:
            classes_idx = torch.arange(B, dtype=torch.int64)
        else:
            classes_idx = torch.zeros((B,), dtype=torch.int64)

            num_classes = 0

            for (i,num_sample_class) in enumerate(self.class_distribution):
                same_class_idx = (classes == i).nonzero().flatten()

                randomized_idx = torch.randperm(same_class_idx.shape[0])[:num_sample_class]

                classes_idx[num_classes:num_classes+num_sample_class] = same_class_idx[randomized_idx]

                num_classes += num_sample_class

        for split_img_idx,idx in enumerate(classes_idx):
            i = idx.item()

            row = i // 8
            col = i % 8

            img = concat_img[:,(row)*W:(row+2)*W,(col+1)*H:(col+2)*H]

            split_img[split_img_idx] = img

        out_classes = torch.gather(classes, 0, classes_idx).type(torch.long)

        return (split_img, out_classes) 

# A simple test
if __name__ == '__main__':
    color_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    depth_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    print(f'Full Image Dataset')

    dset_full = AjedrezDataset('./image_dataset/metadata.csv', './.', 
        use_depth=True, 
        color_transform=color_transforms, 
        depth_transform=depth_transforms,
        full_image=True
    )

    im, c = dset_full[0] 

    print(im.shape, c)

    print(f'Uniform Distribution')

    dset_uniform = AjedrezDataset('./image_dataset/metadata.csv', './.', 
        use_depth=True, 
        color_transform=color_transforms, 
        depth_transform=depth_transforms,
        full_image=False,
        class_distribution=DataUtils.UNIFORM_DISTRIBUTION
    )

    im, c = dset_uniform[0] 

    print(im.shape, c)

    print(f'Limited Distribution')

    dset_modified = AjedrezDataset('./image_dataset/metadata.csv', './.', 
        use_depth=True, 
        color_transform=color_transforms, 
        depth_transform=depth_transforms,
        full_image=False,
        class_distribution=DataUtils.LIMITED_DISTRIBUTION
    )

    im, c = dset_modified[0] 

    print(im.shape, c)

    print(f'Just Pawns')

    dset_modified = AjedrezDataset('./image_dataset/metadata.csv', './.', 
        use_depth=True, 
        color_transform=color_transforms, 
        depth_transform=depth_transforms,
        full_image=False,
        class_distribution=DataUtils.JUST_PAWNS
    )

    im, c = dset_modified[0] 

    print(im.shape, c)