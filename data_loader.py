"""
Data loader for the Potholes dataset with PascalVOC XML annotations.
"""

import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


def parse_voc_xml(xml_path):
    """
    Parse a PascalVOC-style XML annotation file.

    Args:
        xml_path: Path to the XML file

    Returns:
        dict with 'filename', 'width', 'height', and 'boxes' (list of dicts)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Get all bounding boxes
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')

        # PascalVOC uses 1-indexed coordinates
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        boxes.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]  # [x1, y1, x2, y2]
        })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'boxes': boxes
    }


class PotholesDataset(Dataset):
    """
    Dataset class for the Potholes dataset.
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Path to the potholes dataset directory
            split: 'train' or 'test'
            transform: Optional transforms to apply to images
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load splits
        splits_path = os.path.join(data_dir, 'splits.json')
        with open(splits_path, 'r') as f:
            splits = json.load(f)

        self.image_ids = splits[split]

        # Determine image and annotation directories
        # Common structures: images in root or in 'JPEGImages', annotations in 'Annotations'
        self.img_dir = self._find_image_dir()
        self.ann_dir = self._find_annotation_dir()

    def _find_image_dir(self):
        """Find the directory containing images."""
        possible_dirs = [
            self.data_dir,
            os.path.join(self.data_dir, 'JPEGImages'),
            os.path.join(self.data_dir, 'images'),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                # Check if it contains images
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        return d
        return self.data_dir

    def _find_annotation_dir(self):
        """Find the directory containing annotations."""
        possible_dirs = [
            self.data_dir,
            os.path.join(self.data_dir, 'Annotations'),
            os.path.join(self.data_dir, 'annotations'),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                # Check if it contains XML files
                for f in os.listdir(d):
                    if f.lower().endswith('.xml'):
                        return d
        return self.data_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image or transformed tensor
            target: dict with 'boxes' (Nx4 tensor), 'labels' (N tensor), 'image_id'
        """
        image_id = self.image_ids[idx]

        # Load image
        img_path = self._get_image_path(image_id)
        image = Image.open(img_path).convert('RGB')

        # Load annotations
        ann_path = self._get_annotation_path(image_id)
        annotation = parse_voc_xml(ann_path)

        # Extract boxes and labels
        boxes = []
        labels = []
        for obj in annotation['boxes']:
            boxes.append(obj['bbox'])
            labels.append(1)  # 1 for pothole (0 reserved for background)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def _get_image_path(self, image_id):
        """Get the full path to an image."""
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = os.path.join(self.img_dir, image_id + ext)
            if os.path.exists(path):
                return path
        # If image_id already has extension
        path = os.path.join(self.img_dir, image_id)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Image not found for id: {image_id}")

    def _get_annotation_path(self, image_id):
        """Get the full path to an annotation file."""
        # Remove extension if present
        base_id = os.path.splitext(image_id)[0]
        path = os.path.join(self.ann_dir, base_id + '.xml')
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Annotation not found for id: {image_id}")

    def get_image_and_boxes(self, idx):
        """
        Get raw image (as numpy array) and boxes for visualization.

        Returns:
            image: numpy array (H, W, 3)
            boxes: list of [x1, y1, x2, y2]
        """
        image_id = self.image_ids[idx]

        # Load image as numpy
        img_path = self._get_image_path(image_id)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load annotations
        ann_path = self._get_annotation_path(image_id)
        annotation = parse_voc_xml(ann_path)

        boxes = [obj['bbox'] for obj in annotation['boxes']]

        return image, boxes, image_id


def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Split a dataset into training and validation sets.

    Args:
        dataset: PotholesDataset instance
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        train_indices, val_indices
    """
    np.random.seed(seed)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices.tolist(), val_indices.tolist()


if __name__ == "__main__":
    # Test the data loader
    # Update this path to your local data directory
    DATA_DIR = "/dtu/datasets1/02516/potholes"

    # Check if running locally or on HPC
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "./data/potholes"  # Local fallback
        print(f"Using local data directory: {DATA_DIR}")

    if os.path.exists(DATA_DIR):
        dataset = PotholesDataset(DATA_DIR, split='train')
        print(f"Loaded {len(dataset)} training images")

        # Test loading one sample
        image, target = dataset[0]
        print(f"Image size: {image.size}")
        print(f"Number of boxes: {len(target['boxes'])}")
        print(f"Boxes: {target['boxes']}")
    else:
        print(f"Data directory not found: {DATA_DIR}")
        print("Please download or sync the data first.")
