import torch
import numpy as np
from PIL import Image 

from torchvision import transforms

IMAGE_SIZE = 640

DEFAULT_DISTRIBUTION = [
    32,
    8, 2, 2, 2, 1, 1,
    8, 2, 2, 2, 1, 1
]

LIMITED_DISTRIBUTION = [
    2,
    2, 2, 2, 2, 1, 1,
    2, 2, 2, 2, 1, 1
]

JUST_PAWNS = [
    0,
    4, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0
]

UNIFORM_DISTRIBUTION = [1] * 13

def split_image_by_classes(color_image, depth_image, class_distribution, classes):
    assert len(class_distribution) == 13
    assert len(classes) == 64
    assert isinstance(color_image, np.ndarray)

    classes = np.array(classes)

    B = sum(class_distribution)
    H = IMAGE_SIZE // 10
    W = IMAGE_SIZE // 10
    C = 4

    concat_images = np.zeros((B, 2*H, W, C), dtype=np.uint8)

    num_classes = 0

    classes_idx = np.zeros((B,))

    for i,num_sample_class in enumerate(class_distribution):
        same_class_idx = (classes == i).nonzero()[0].flatten()

        randomized_idx = np.random.choice(same_class_idx, size=num_sample_class, replace=False) 

        classes_idx[num_classes:num_classes+num_sample_class] = randomized_idx

        num_classes += num_sample_class

    for split_img_idx,idx in enumerate(classes_idx.astype(np.uint8)):
        row = idx // 8
        col = idx % 8

        cimg = color_image[(row)*W:(row+2)*W,(col+1)*H:(col+2)*H,:]
        dimg = depth_image[(row)*W:(row+2)*W,(col+1)*H:(col+2)*H]

        concat_images[split_img_idx,:,:,:3] = cimg
        concat_images[split_img_idx,:,:,3] = dimg

    return (concat_images, classes[classes_idx.astype(np.uint8)])


def split_image_pytorch(color_image, transforms, imsize=224):
    B = 64
    H = IMAGE_SIZE // 10 
    W = IMAGE_SIZE // 10
    C = 3 

    concat_images = torch.zeros((B, C, imsize, imsize))

    for b in range(B):
        row = b // 8
        col = b % 8

        cropped_img = color_image[(row)*W:(row+2)*W,(col+1)*H:(col+2)*H,:] 
        cropped_img = Image.fromarray(cropped_img, 'RGB')
        concat_images[b] = transforms(cropped_img)             

    return concat_images