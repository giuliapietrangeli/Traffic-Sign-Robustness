import torch
import torch.nn as nn
from data_loader import get_dataloaders, get_corrupted_test_loader
from model import get_model
import argparse
import os
import numpy as np

def evaluate(model_path, device='cuda'):
    print(f"Evaluating model: {model_path}")
    
    # Load model
    model = get_model(num_classes=43).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    # Clean Test Set
    _, _, test_loader = get_dataloaders(batch_size=64, augment_train=False)
    
    clean_acc = evaluate_loader(model, test_loader, device)
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    
    # Corrupted Test Sets
    corruptions = ['rain', 'fog', 'blur']
    results = {'Clean': clean_acc}
    
    for corruption in corruptions:
        loader = get_corrupted_test_loader(corruption=corruption)
        acc = evaluate_loader(model, loader, device)
        print(f"{corruption.capitalize()} Test Accuracy: {acc:.2f}%")
        results[corruption.capitalize()] = acc
        
    return results

def evaluate_loader(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    evaluate(args.model_path, device)
