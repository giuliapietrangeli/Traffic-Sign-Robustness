import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_transforms
from torchvision.datasets import GTSRB
import os

def visualize_augmentations(output_path='augmentation_samples.png'):
    # Load a few images
    dataset = GTSRB(root='data', split='train', download=False)
    
    # Pick random indices
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    # Augmentations to test
    augmentations = {
        'Original': None,
        'Rain': 'rain',
        'Fog': 'fog',
        'Blur': 'blur',
        'Random': 'random'
    }
    
    fig, axes = plt.subplots(len(indices), len(augmentations), figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx] # PIL Image
        
        for j, (name, aug_type) in enumerate(augmentations.items()):
            if name == 'Original':
                # Just resize for visualization
                img_vis = img.resize((32, 32))
                axes[i, j].imshow(img_vis)
            else:
                if name == 'Random':
                    transform = get_transforms(augment=True)
                else:
                    transform = get_transforms(specific_corruption=aug_type)
                
                # Apply transform
                # The transform returns a normalized tensor.
                tensor = transform(img)
                
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = tensor.permute(1, 2, 0).numpy() * std + mean
                img_vis = np.clip(img_vis, 0, 1)
                
                axes[i, j].imshow(img_vis)
            
            if i == 0:
                axes[i, j].set_title(name)
            axes[i, j].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_augmentations()
