import os
import pandas as pd
import numpy as np
import cv2
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.transforms import ToTensor
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import transforms
# from torchinfo import summary
import timm
import segmentation_models_pytorch as smp
import argparse

model = smp.UnetPlusPlus(
    encoder_name="resnet101",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3,
    decoder_attention_type='scse'
)

train_transform = A.Compose(
    [
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.VerticalFlip(),
    A.HorizontalFlip(),
    A.OneOf([
    A.Blur(blur_limit=3, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.MedianBlur(blur_limit=3, p=0.3),
    A.RandomBrightnessContrast()
    ], p=0.3),
    A.Rotate(limit=30, p=0.3),
    A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2(), 
]
)

val_transform = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting image input")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    checkpoint = torch.load('model/segmentation_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainsize = 256
    model.eval()
    img_path = args.image_path
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (trainsize, trainsize))
    transformed = val_transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"output.jpg", mask_rgb)
