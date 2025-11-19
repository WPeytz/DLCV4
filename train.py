"""
Training and evaluation functions for the proposal classifier.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=0):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        avg_loss: Average loss
        accuracy: Validation accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs=10, save_path='best_model.pth', patience=5):
    """
    Full training loop with validation and early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs
        save_path: Path to save best model
        patience: Early stopping patience

    Returns:
        history: Dict with training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  Saved best model (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    return history


def print_evaluation_report(preds, labels, class_names=['background', 'pothole']):
    """
    Print detailed evaluation metrics.

    Args:
        preds: Predicted labels
        labels: True labels
        class_names: Names for each class
    """
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(labels, preds, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(f"              Predicted")
    print(f"              {class_names[0]:>10} {class_names[1]:>10}")
    print(f"Actual {class_names[0]:>10} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"       {class_names[1]:>10} {cm[1,0]:>10} {cm[1,1]:>10}")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {name}: {class_acc:.4f}")


def plot_training_history(history):
    """
    Plot training history.

    Args:
        history: Dict with training history

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_optimizer(model, lr=0.001, weight_decay=1e-4):
    """
    Create optimizer with different learning rates for backbone and classifier.

    Args:
        model: PyTorch model
        lr: Base learning rate
        weight_decay: Weight decay

    Returns:
        optimizer
    """
    # Get parameters from different parts of the model
    if hasattr(model, 'backbone'):
        # Lower learning rate for pretrained backbone
        backbone_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': classifier_params, 'lr': lr}
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def get_scheduler(optimizer, patience=3, factor=0.5):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        patience: Number of epochs to wait before reducing LR
        factor: Factor to reduce LR by

    Returns:
        scheduler
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=factor
    )


if __name__ == "__main__":
    # Quick test
    print("Training utilities loaded successfully")
