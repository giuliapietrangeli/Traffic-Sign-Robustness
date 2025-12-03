import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.model_selection import train_test_split

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # img is PIL, convert to numpy
        img = np.array(img)
        return self.transform(image=img)['image']

def get_transforms(augment=False, specific_corruption=None):
    """
    augment: If True, applies random weather augmentations (for training robust model).
    specific_corruption: If set (e.g., 'rain', 'fog', 'blur'), applies ONLY that corruption (for evaluation).
    """
    transforms_list = [
        A.Resize(32, 32),
    ]

    if specific_corruption:
        if specific_corruption == 'rain':
            transforms_list.append(A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0))
        elif specific_corruption == 'fog':
            transforms_list.append(A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1.0))
        elif specific_corruption == 'blur':
            transforms_list.append(A.MotionBlur(blur_limit=5, p=1.0))
    elif augment:
        # Randomly apply one of the weather effects during training
        transforms_list.append(
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
                A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.5) # 50% chance of corruption
        )

    transforms_list.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(A.Compose(transforms_list))

def get_dataloaders(data_dir='data', batch_size=64, val_split=0.2, num_workers=2, augment_train=False):
    """
    augment_train: If True, applies random weather augmentations to the training set.
    """
    # Train set transform
    train_transforms = get_transforms(augment=augment_train)
    # Validation/Test set transform (clean)
    val_transforms = get_transforms(augment=False)

    # Load dataset for splitting
    # Note: We use the same dataset object but will wrap it or load it twice to apply different transforms
    # To ensure same split, we fix random_state
    
    # We load the dataset without transforms first to get labels for stratification
    # Actually GTSRB loads images on the fly, so we can just load it.
    # But we need labels. GTSRB has ._samples which is a list of (path, class_index)
    
    # Let's just load it once to get indices
    temp_dataset = GTSRB(root=data_dir, split='train', download=False)
    labels = [y for _, y in temp_dataset]
    
    train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=val_split, stratify=labels, random_state=42)
    
    # Now load with correct transforms
    train_dataset = GTSRB(root=data_dir, split='train', transform=train_transforms, download=False)
    val_dataset = GTSRB(root=data_dir, split='train', transform=val_transforms, download=False)
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    test_dataset = GTSRB(root=data_dir, split='test', transform=val_transforms, download=False)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def get_corrupted_test_loader(data_dir='data', batch_size=64, corruption='rain', num_workers=2):
    transform = get_transforms(specific_corruption=corruption)
    test_dataset = GTSRB(root=data_dir, split='test', transform=transform, download=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
