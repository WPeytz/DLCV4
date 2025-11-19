"""
Dataset and DataLoader for training the object detector.
Handles class imbalance between positive (pothole) and negative (background) proposals.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms


class ProposalDataset(Dataset):
    """
    Dataset for training proposal classifier.
    Crops proposals from images and returns them with labels.
    """

    def __init__(self, training_data, data_dir, image_ids, transform=None,
                 neg_pos_ratio=3, filter_ignored=True):
        """
        Args:
            training_data: Dict from prepare_training_data() with proposals and labels
            data_dir: Path to the potholes dataset
            image_ids: List of image IDs to include (for train/val split)
            transform: Torchvision transforms for data augmentation
            neg_pos_ratio: Maximum ratio of negatives to positives per image
            filter_ignored: Whether to filter out ignored proposals (label=-1)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.neg_pos_ratio = neg_pos_ratio

        # Find image directory
        self.img_dir = self._find_image_dir()

        # Build list of (image_id, proposal_idx, box, label) tuples
        self.samples = []

        for image_id in image_ids:
            if image_id not in training_data:
                continue

            data = training_data[image_id]
            proposals = data['proposals']
            labels = data['labels']

            # Separate positives and negatives
            pos_indices = [i for i, l in enumerate(labels) if l == 1]
            neg_indices = [i for i, l in enumerate(labels) if l == 0]

            # Add all positives
            for idx in pos_indices:
                self.samples.append((image_id, idx, proposals[idx], 1))

            # Limit negatives based on neg_pos_ratio
            max_neg = max(len(pos_indices) * neg_pos_ratio, 10)  # At least 10 negatives
            if len(neg_indices) > max_neg:
                neg_indices = np.random.choice(neg_indices, max_neg, replace=False).tolist()

            for idx in neg_indices:
                self.samples.append((image_id, idx, proposals[idx], 0))

        # Count samples per class
        self.num_pos = sum(1 for s in self.samples if s[3] == 1)
        self.num_neg = sum(1 for s in self.samples if s[3] == 0)

        print(f"Dataset created: {len(self.samples)} samples "
              f"({self.num_pos} positive, {self.num_neg} negative)")

    def _find_image_dir(self):
        """Find the directory containing images."""
        possible_dirs = [
            os.path.join(self.data_dir, 'images'),
            os.path.join(self.data_dir, 'JPEGImages'),
            self.data_dir,
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                return d
        return self.data_dir

    def _get_image_path(self, image_id):
        """Get full path to image."""
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            path = os.path.join(self.img_dir, image_id + ext)
            if os.path.exists(path):
                return path
        # Try with extension already included
        path = os.path.join(self.img_dir, image_id)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Image not found: {image_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, prop_idx, box, label = self.samples[idx]

        # Load image
        img_path = self._get_image_path(image_id)
        image = Image.open(img_path).convert('RGB')

        # Crop proposal region
        x1, y1, x2, y2 = box
        # Ensure valid coordinates
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))

        crop = image.crop((x1, y1, x2, y2))

        # Apply transforms
        if self.transform:
            crop = self.transform(crop)

        return crop, label

    def get_class_weights(self):
        """
        Calculate class weights for weighted loss.
        Returns weights inversely proportional to class frequency.
        """
        total = len(self.samples)
        weights = torch.tensor([
            total / (2 * self.num_neg),  # Weight for class 0 (background)
            total / (2 * self.num_pos)   # Weight for class 1 (pothole)
        ], dtype=torch.float32)
        return weights

    def get_sample_weights(self):
        """
        Get per-sample weights for WeightedRandomSampler.
        This helps balance mini-batches during training.
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[s[3]].item() for s in self.samples]
        return sample_weights


def get_transforms(split='train', input_size=224):
    """
    Get transforms for training or validation.

    Args:
        split: 'train' or 'val'
        input_size: Size to resize images to

    Returns:
        torchvision.transforms.Compose
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_data_loaders(training_data, data_dir, train_ids, val_ids,
                        batch_size=32, num_workers=4, neg_pos_ratio=3,
                        use_weighted_sampler=True):
    """
    Create training and validation data loaders.

    Args:
        training_data: Dict from prepare_training_data()
        data_dir: Path to potholes dataset
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        batch_size: Batch size
        num_workers: Number of data loading workers
        neg_pos_ratio: Max ratio of negatives to positives
        use_weighted_sampler: Use weighted sampling to balance batches

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = ProposalDataset(
        training_data, data_dir, train_ids,
        transform=get_transforms('train'),
        neg_pos_ratio=neg_pos_ratio
    )

    val_dataset = ProposalDataset(
        training_data, data_dir, val_ids,
        transform=get_transforms('val'),
        neg_pos_ratio=neg_pos_ratio * 2  # More negatives for thorough validation
    )

    # Create samplers
    if use_weighted_sampler:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_train_val_split(image_ids, val_ratio=0.2, seed=42):
    """
    Split image IDs into training and validation sets.

    Args:
        image_ids: List of image IDs
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        train_ids, val_ids
    """
    np.random.seed(seed)
    ids = np.array(image_ids)
    np.random.shuffle(ids)

    n_val = int(len(ids) * val_ratio)
    val_ids = ids[:n_val].tolist()
    train_ids = ids[n_val:].tolist()

    return train_ids, val_ids


if __name__ == "__main__":
    # Test the dataset
    import json

    DATA_DIR = "./potholes"

    # Load training data from Part 1
    if os.path.exists("training_data.npy"):
        training_data = np.load("training_data.npy", allow_pickle=True).item()
        print(f"Loaded training data for {len(training_data)} images")

        # Get image IDs
        image_ids = list(training_data.keys())
        train_ids, val_ids = create_train_val_split(image_ids)
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

        # Create dataset
        dataset = ProposalDataset(
            training_data, DATA_DIR, train_ids[:10],  # Test with first 10
            transform=get_transforms('train'),
            neg_pos_ratio=3
        )

        # Test loading a sample
        crop, label = dataset[0]
        print(f"Sample shape: {crop.shape}, label: {label}")

        # Test class weights
        weights = dataset.get_class_weights()
        print(f"Class weights: {weights}")
    else:
        print("training_data.npy not found. Run Part 1 first.")
