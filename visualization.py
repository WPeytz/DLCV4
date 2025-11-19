"""
Visualization utilities for the Potholes dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def plot_image_with_boxes(image, boxes, labels=None, scores=None,
                          title=None, ax=None, color='green', linewidth=2):
    """
    Plot an image with bounding boxes.

    Args:
        image: numpy array (H, W, 3) or PIL Image
        boxes: list of [x1, y1, x2, y2] or tensor of shape (N, 4)
        labels: optional list of labels for each box
        scores: optional list of confidence scores
        title: optional title for the plot
        ax: matplotlib axis (creates new figure if None)
        color: box color
        linewidth: box line width

    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 8))

    # Convert to numpy if needed
    if hasattr(image, 'numpy'):
        image = image.numpy()
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Handle tensor format (C, H, W) -> (H, W, C)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)

    ax.imshow(image)

    # Convert boxes to list if tensor
    if hasattr(boxes, 'numpy'):
        boxes = boxes.numpy()
    if hasattr(boxes, 'tolist'):
        boxes = boxes.tolist()

    # Draw each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label/score if provided
        label_text = ""
        if labels is not None and i < len(labels):
            label_text = str(labels[i])
        if scores is not None and i < len(scores):
            score_text = f"{scores[i]:.2f}"
            label_text = f"{label_text} {score_text}" if label_text else score_text

        if label_text:
            ax.text(x1, y1 - 5, label_text,
                    color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    if title:
        ax.set_title(title)
    ax.axis('off')

    return ax


def plot_proposals_and_gt(image, proposals, gt_boxes,
                          max_proposals=100, title=None):
    """
    Plot image with both proposals and ground truth boxes.

    Args:
        image: numpy array (H, W, 3)
        proposals: list of [x1, y1, x2, y2] proposal boxes
        gt_boxes: list of [x1, y1, x2, y2] ground truth boxes
        max_proposals: maximum number of proposals to display
        title: optional title

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(1, figsize=(14, 10))

    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    ax.imshow(image)

    # Draw proposals (blue, thin)
    for i, box in enumerate(proposals[:max_proposals]):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1,
            edgecolor='blue',
            facecolor='none',
            alpha=0.3
        )
        ax.add_patch(rect)

    # Draw ground truth (green, thick)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Proposals (blue): {min(len(proposals), max_proposals)}, "
                     f"GT boxes (green): {len(gt_boxes)}")
    ax.axis('off')

    return fig, ax


def plot_dataset_samples(dataset, num_samples=6, cols=3):
    """
    Plot multiple samples from the dataset with their ground truth boxes.

    Args:
        dataset: PotholesDataset instance
        num_samples: number of samples to plot
        cols: number of columns in the grid
    """
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        if i < len(dataset):
            image, boxes, image_id = dataset.get_image_and_boxes(i)
            plot_image_with_boxes(
                image, boxes,
                title=f"{image_id}\n({len(boxes)} potholes)",
                ax=axes[i]
            )
        else:
            axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_recall_vs_proposals(recalls, proposal_counts, title="Recall vs Number of Proposals"):
    """
    Plot recall curve as a function of number of proposals.

    Args:
        recalls: list of recall values
        proposal_counts: list of corresponding proposal counts
        title: plot title

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(1, figsize=(10, 6))

    ax.plot(proposal_counts, recalls, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Proposals', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add horizontal line at common thresholds
    for threshold in [0.5, 0.75, 0.9]:
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        ax.text(proposal_counts[-1] * 0.02, threshold + 0.02,
                f'{threshold:.0%}', fontsize=10, color='gray')

    return fig, ax


def plot_iou_distribution(ious, title="IoU Distribution"):
    """
    Plot histogram of IoU values.

    Args:
        ious: list of IoU values
        title: plot title

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(1, figsize=(10, 6))

    ax.hist(ious, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='IoU=0.5 threshold')
    ax.set_xlabel('IoU', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


if __name__ == "__main__":
    # Test visualization with dummy data
    import numpy as np

    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Dummy boxes
    boxes = [
        [100, 100, 200, 200],
        [300, 150, 450, 300],
    ]

    fig, ax = plt.subplots(1, figsize=(10, 8))
    plot_image_with_boxes(image, boxes, title="Test Visualization")
    plt.show()
