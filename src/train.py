import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import get_model
import argparse
import os
from tqdm import tqdm

def train(augment=False, epochs=10, batch_size=64, lr=0.001, device='cuda'):
    print(f"Training with Augmentation: {augment}")
    
    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, augment_train=augment)
    
    # Model
    model = get_model(num_classes=43).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_acc = 0.0
    save_name = 'model_robust.pth' if augment else 'model_baseline.pth'
    
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update pbar description with current loss/acc
            pbar.set_postfix({'loss': f"{running_loss/len(train_loader):.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join('models', save_name))
            print(f"Saved best model to models/{save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true', help='Train with data augmentation')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    train(augment=args.augment, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
