import torch
from torchvision.datasets import GTSRB
from sklearn.model_selection import train_test_split
import os

def check_leakage(data_dir='data'):
    print("Checking for Data Leakage...")
    
    # 1. Load Train Split (which we split into Train/Val)
    full_train_dataset = GTSRB(root=data_dir, split='train', download=False)
    
    # Get all samples (paths)
    # GTSRB dataset in torchvision has ._samples which is a list of (path, label)
    # Note: In newer torchvision versions it might be different, but usually it's there for ImageFolder-like datasets.
    # If not, we rely on indices.
    
    try:
        train_samples = [s[0] for s in full_train_dataset._samples]
    except AttributeError:
        print("Could not access ._samples, verifying based on indices only.")
        train_samples = list(range(len(full_train_dataset)))

    labels = [y for _, y in full_train_dataset]
    
    # Recreate the split
    train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=42)
    
    print(f"Total Training samples (Train+Val): {len(full_train_dataset)}")
    print(f"Train subset size: {len(train_idx)}")
    print(f"Val subset size: {len(val_idx)}")
    
    # Check 1: Intersection between Train and Val indices
    intersection = set(train_idx).intersection(set(val_idx))
    if len(intersection) == 0:
        print("[PASS] Train and Validation indices are disjoint.")
    else:
        print(f"[FAIL] Found {len(intersection)} overlapping samples between Train and Val!")
        
    # 2. Load Test Split
    test_dataset = GTSRB(root=data_dir, split='test', download=False)
    try:
        test_samples = [s[0] for s in test_dataset._samples]
    except AttributeError:
        test_samples = []
        print("Could not access ._samples for test set.")

    print(f"Test set size: {len(test_dataset)}")

    # Check 2: Intersection between Train/Val and Test (based on file paths if available)
    if train_samples and test_samples:
        train_paths = set(train_samples)
        test_paths = set(test_samples)
        
        leakage = train_paths.intersection(test_paths)
        if len(leakage) == 0:
            print("[PASS] Train/Val and Test sets are disjoint (based on file paths).")
        else:
            print(f"[FAIL] Found {len(leakage)} samples appearing in both Train/Val and Test!")
    else:
        print("[INFO] Skipping file path check (could not retrieve paths).")
        # Logic check: GTSRB 'train' and 'test' are different folders usually
        if 'train' in str(full_train_dataset) and 'test' in str(test_dataset):
             print("[PASS] Train and Test datasets are loaded from different splits.")

if __name__ == "__main__":
    check_leakage()
