"""
Object proposal extraction using Selective Search.
"""

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def selective_search(image, mode='fast', resize_height=None):
    """
    Run Selective Search on an image to generate object proposals.

    Args:
        image: numpy array (H, W, 3) in RGB format or PIL Image
        mode: 'fast' or 'quality' (quality generates more proposals)
        resize_height: optionally resize image for faster processing

    Returns:
        boxes: list of [x1, y1, x2, y2] proposals
        scale: scaling factor used (for mapping back to original size)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Store original size
    orig_height, orig_width = image.shape[:2]
    scale = 1.0

    # Resize for efficiency if specified
    if resize_height is not None and orig_height > resize_height:
        scale = resize_height / orig_height
        new_width = int(orig_width * scale)
        image = cv2.resize(image, (new_width, resize_height))

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)

    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    # Run selective search
    rects = ss.process()

    # Convert from (x, y, w, h) to (x1, y1, x2, y2) and scale back
    boxes = []
    for (x, y, w, h) in rects:
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + w) / scale)
        y2 = int((y + h) / scale)

        # Clip to image bounds
        x1 = max(0, min(x1, orig_width))
        y1 = max(0, min(y1, orig_height))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))

        if x2 > x1 and y2 > y1:  # Valid box
            boxes.append([x1, y1, x2, y2])

    return boxes


def extract_proposals_for_dataset(dataset, mode='fast', resize_height=480,
                                  max_proposals=2000, save_path=None):
    """
    Extract proposals for all images in a dataset.

    Args:
        dataset: PotholesDataset instance
        mode: 'fast' or 'quality'
        resize_height: resize images to this height for SS
        max_proposals: maximum proposals to keep per image
        save_path: optional path to save proposals as .npy

    Returns:
        all_proposals: dict mapping image_id to list of proposals
    """
    all_proposals = {}

    print(f"Extracting proposals for {len(dataset)} images...")

    for idx in tqdm(range(len(dataset))):
        image, boxes, image_id = dataset.get_image_and_boxes(idx)

        # Run selective search
        proposals = selective_search(image, mode=mode, resize_height=resize_height)

        # Limit number of proposals
        proposals = proposals[:max_proposals]

        all_proposals[image_id] = proposals

    if save_path:
        np.save(save_path, all_proposals)
        print(f"Saved proposals to {save_path}")

    return all_proposals


def load_proposals(path):
    """Load proposals from a .npy file."""
    return np.load(path, allow_pickle=True).item()


def filter_proposals_by_size(proposals, min_size=10, max_aspect_ratio=5):
    """
    Filter proposals by size and aspect ratio.

    Args:
        proposals: list of [x1, y1, x2, y2]
        min_size: minimum width/height
        max_aspect_ratio: maximum aspect ratio (w/h or h/w)

    Returns:
        filtered proposals
    """
    filtered = []
    for box in proposals:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if w < min_size or h < min_size:
            continue

        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > max_aspect_ratio:
            continue

        filtered.append(box)

    return filtered


def get_proposal_statistics(all_proposals):
    """
    Get statistics about extracted proposals.

    Args:
        all_proposals: dict mapping image_id to proposals

    Returns:
        stats dict
    """
    counts = [len(props) for props in all_proposals.values()]

    stats = {
        'num_images': len(all_proposals),
        'total_proposals': sum(counts),
        'mean_per_image': np.mean(counts),
        'std_per_image': np.std(counts),
        'min_per_image': min(counts),
        'max_per_image': max(counts),
    }

    return stats


if __name__ == "__main__":
    # Test with a dummy image
    print("Testing Selective Search...")

    # Create a test image with some structure
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some rectangles to give SS something to find
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(image, (300, 150), (450, 300), (0, 255, 0), -1)
    cv2.rectangle(image, (50, 350), (150, 450), (0, 0, 255), -1)

    # Run selective search
    proposals = selective_search(image, mode='fast')
    print(f"Generated {len(proposals)} proposals")

    # Filter proposals
    filtered = filter_proposals_by_size(proposals)
    print(f"After filtering: {len(filtered)} proposals")
