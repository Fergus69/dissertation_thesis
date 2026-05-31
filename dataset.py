import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch


train_transform = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),a
    A.Rotate(limit=10, p=0.5),
    A.ImageCompression(quality_range=(60, 90), p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.CenterCrop(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class AIDetectionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, img_path
    
class TTADetectionDataset(Dataset):
    def __init__(self, image_paths, labels, crop_size=384):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        
        
        self.pad_if_needed = A.PadIfNeeded(
            min_height=crop_size, 
            min_width=crop_size, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        
        self.normalize = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        image = self.pad_if_needed(image=image)['image']
        
        h, w, _ = image.shape
        c = self.crop_size
        
        
        crops = [
            image[0:c, 0:c],                                
            image[0:c, w-c:w],                              
            image[h-c:h, 0:c],                              
            image[h-c:h, w-c:w],                            
            image[(h-c)//2:(h+c)//2, (w-c)//2:(w+c)//2]     
        ]
        
        
        tensor_crops = []
        for crop in crops:
            tensor_crops.append(self.normalize(image=crop)['image'])
            
        
        stacked_crops = torch.stack(tensor_crops)
        label = self.labels[idx]

        return stacked_crops, label, img_path
