"""
Project 4.2 - Object Detector

This script implements the training and evaluation of a CNN classifier for pothole detection.

Tasks:
1. Build a CNN to classify object proposals (2 classes: pothole + background)
2. Build a dataloader with class imbalance handling
3. Finetune the network on the training set
4. Evaluate classification accuracy on the validation set
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Local modules
from model import ProposalClassifier, create_model
from detector_dataset import (
    ProposalDataset,
    create_data_loaders,
    create_train_val_split,
    get_transforms
)
from train import (
    train_model,
    evaluate,
    print_evaluation_report,
    plot_training_history,
    get_optimizer,
    get_scheduler
)


def main():
    plt.rcParams['figure.figsize'] = [12, 8]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    DATA_DIR = "./potholes"
    TRAINING_DATA_PATH = "training_data.npy"

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    NEG_POS_RATIO = 3  # Max ratio of negatives to positives
    VAL_RATIO = 0.2

    # Check if training data exists
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}\n"
            "Please run Part 1 (main.py) first to generate training data."
        )

    print(f"Configuration loaded")

    # ==========================================================================
    # Task 1: Build CNN Classifier
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 1: Build CNN Classifier")
    print("=" * 60)

    # Create model
    model = create_model(
        model_type='resnet18',
        num_classes=2,
        pretrained=True,
        freeze_backbone=False
    )

    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    test_input = torch.randn(4, 3, 224, 224).to(device)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output (logits): {test_output}")

    # ==========================================================================
    # Task 2: Build DataLoader
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 2: Build DataLoader")
    print("=" * 60)

    # Load training data from Part 1
    training_data = np.load(TRAINING_DATA_PATH, allow_pickle=True).item()
    print(f"Loaded training data for {len(training_data)} images")

    # Get image IDs and create train/val split
    image_ids = list(training_data.keys())
    train_ids, val_ids = create_train_val_split(image_ids, val_ratio=VAL_RATIO)
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        training_data,
        DATA_DIR,
        train_ids,
        val_ids,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Set to 0 for debugging, increase for speed
        neg_pos_ratio=NEG_POS_RATIO,
        use_weighted_sampler=True
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Visualize some training samples
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(8):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        label_name = 'Pothole' if labels[i] == 1 else 'Background'
        color = 'green' if labels[i] == 1 else 'red'
        axes[i].set_title(label_name, color=color)
        axes[i].axis('off')

    plt.suptitle('Training Samples (Cropped Proposals)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Get class weights for weighted loss
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    print(f"  Background weight: {class_weights[0]:.4f}")
    print(f"  Pothole weight: {class_weights[1]:.4f}")

    # ==========================================================================
    # Task 3: Finetune the Network
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 3: Finetune the Network")
    print("=" * 60)

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, patience=3, factor=0.5)

    print("Training setup complete")
    print(f"  Loss: CrossEntropyLoss with class weights")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Scheduler: ReduceLROnPlateau")

    # Train the model
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=NUM_EPOCHS,
        save_path='best_model.pth',
        patience=5
    )

    # Plot training history
    fig = plot_training_history(history)
    plt.show()

    print(f"\nBest validation accuracy: {max(history['val_acc']):.4f}")

    # ==========================================================================
    # Task 4: Evaluate Classification Accuracy
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 4: Evaluate Classification Accuracy")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")

    # Evaluate on validation set
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)

    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")

    # Detailed evaluation report
    print_evaluation_report(preds, labels, class_names=['background', 'pothole'])

    # Visualize confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Pothole'],
                yticklabels=['Background', 'Pothole'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Visualize some predictions
    model.eval()
    images, true_labels = next(iter(val_loader))

    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, pred_labels = outputs.max(1)

    # Plot predictions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(8):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        pred = pred_labels[i].item()
        true = true_labels[i].item()
        prob = probs[i, pred].item()

        pred_name = 'Pothole' if pred == 1 else 'Background'
        true_name = 'Pothole' if true == 1 else 'Background'

        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred_name} ({prob:.2f})\nTrue: {true_name}', color=color)
        axes[i].axis('off')

    plt.suptitle('Validation Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
In this script, we:

1. Built a CNN classifier: ResNet18 backbone with custom classification head
2. Created balanced data loaders: Used weighted sampling and limited neg/pos ratio
3. Trained the network: Fine-tuned on the pothole dataset
4. Evaluated accuracy: Measured classification performance on validation set

Note:
- This is classification accuracy on proposals, not object detection mAP
- The model can now be used in Part 4.3 for full detection pipeline
""")


if __name__ == "__main__":
    main()
