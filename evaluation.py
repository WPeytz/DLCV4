"""
Evaluation functions for object proposals.
Includes IoU calculation, recall computation, and label assignment.
"""

import numpy as np
from tqdm import tqdm


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        iou: float between 0 and 1
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)

    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_matrix(proposals, gt_boxes):
    """
    Compute IoU matrix between proposals and ground truth boxes.

    Args:
        proposals: list of N proposals [x1, y1, x2, y2]
        gt_boxes: list of M ground truth boxes [x1, y1, x2, y2]

    Returns:
        iou_matrix: numpy array of shape (N, M)
    """
    n_proposals = len(proposals)
    n_gt = len(gt_boxes)

    if n_proposals == 0 or n_gt == 0:
        return np.zeros((n_proposals, n_gt))

    iou_matrix = np.zeros((n_proposals, n_gt))

    for i, prop in enumerate(proposals):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(prop, gt)

    return iou_matrix


def compute_recall_at_k(proposals, gt_boxes, k, iou_threshold=0.5):
    """
    Compute recall using top-k proposals.

    Args:
        proposals: list of proposals for one image
        gt_boxes: list of ground truth boxes for one image
        k: number of proposals to use
        iou_threshold: IoU threshold for positive match

    Returns:
        recall: float between 0 and 1
    """
    if len(gt_boxes) == 0:
        return 1.0  # No ground truth = perfect recall

    if len(proposals) == 0:
        return 0.0

    # Use only top-k proposals
    proposals_k = proposals[:k]

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(proposals_k, gt_boxes)

    # For each GT box, check if any proposal has IoU >= threshold
    max_ious = iou_matrix.max(axis=0) if iou_matrix.size > 0 else np.array([])
    detected = (max_ious >= iou_threshold).sum()

    return detected / len(gt_boxes)


def evaluate_proposals(all_proposals, dataset, proposal_counts=[10, 50, 100, 200, 500, 1000, 2000],
                       iou_threshold=0.5):
    """
    Evaluate proposals on a dataset for different numbers of proposals.

    Args:
        all_proposals: dict mapping image_id to proposals
        dataset: PotholesDataset instance
        proposal_counts: list of k values to evaluate
        iou_threshold: IoU threshold for positive match

    Returns:
        results: dict with 'proposal_counts' and 'recalls'
    """
    recalls_per_k = {k: [] for k in proposal_counts}

    print(f"Evaluating proposals at IoU threshold {iou_threshold}...")

    for idx in tqdm(range(len(dataset))):
        _, gt_boxes, image_id = dataset.get_image_and_boxes(idx)

        if image_id not in all_proposals:
            print(f"Warning: No proposals for {image_id}")
            continue

        proposals = all_proposals[image_id]

        for k in proposal_counts:
            recall = compute_recall_at_k(proposals, gt_boxes, k, iou_threshold)
            recalls_per_k[k].append(recall)

    # Compute mean recall for each k
    mean_recalls = [np.mean(recalls_per_k[k]) for k in proposal_counts]

    return {
        'proposal_counts': proposal_counts,
        'recalls': mean_recalls,
        'recalls_per_image': recalls_per_k
    }


def assign_labels_to_proposals(proposals, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    """
    Assign labels to proposals based on IoU with ground truth.

    Args:
        proposals: list of proposals [x1, y1, x2, y2]
        gt_boxes: list of ground truth boxes
        pos_iou_threshold: IoU >= this is positive
        neg_iou_threshold: IoU < this is negative

    Returns:
        labels: list of labels (1 for positive, 0 for negative, -1 for ignore)
        max_ious: list of max IoU for each proposal
        matched_gt: list of matched GT index for each proposal (-1 if none)
    """
    n_proposals = len(proposals)

    if n_proposals == 0:
        return [], [], []

    if len(gt_boxes) == 0:
        # All proposals are background
        return [0] * n_proposals, [0.0] * n_proposals, [-1] * n_proposals

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(proposals, gt_boxes)

    # Get max IoU and matched GT for each proposal
    max_ious = iou_matrix.max(axis=1)
    matched_gt = iou_matrix.argmax(axis=1)

    # Assign labels
    labels = []
    for i in range(n_proposals):
        if max_ious[i] >= pos_iou_threshold:
            labels.append(1)  # Positive (pothole)
        elif max_ious[i] < neg_iou_threshold:
            labels.append(0)  # Negative (background)
        else:
            labels.append(-1)  # Ignore (ambiguous)

    return labels, max_ious.tolist(), matched_gt.tolist()


def prepare_training_data(all_proposals, dataset, pos_iou_threshold=0.5,
                          neg_iou_threshold=0.3, max_proposals_per_image=None):
    """
    Prepare proposals with labels for training an object detector.

    Args:
        all_proposals: dict mapping image_id to proposals
        dataset: PotholesDataset instance
        pos_iou_threshold: IoU threshold for positive labels
        neg_iou_threshold: IoU threshold for negative labels
        max_proposals_per_image: limit proposals per image

    Returns:
        training_data: dict mapping image_id to {
            'proposals': list of boxes,
            'labels': list of labels,
            'ious': list of max IoU values
        }
    """
    training_data = {}

    print("Preparing training data with labels...")

    for idx in tqdm(range(len(dataset))):
        _, gt_boxes, image_id = dataset.get_image_and_boxes(idx)

        if image_id not in all_proposals:
            continue

        proposals = all_proposals[image_id]

        # Limit proposals if specified
        if max_proposals_per_image:
            proposals = proposals[:max_proposals_per_image]

        # Assign labels
        labels, ious, _ = assign_labels_to_proposals(
            proposals, gt_boxes,
            pos_iou_threshold=pos_iou_threshold,
            neg_iou_threshold=neg_iou_threshold
        )

        training_data[image_id] = {
            'proposals': proposals,
            'labels': labels,
            'ious': ious
        }

    # Print statistics
    all_labels = []
    for data in training_data.values():
        all_labels.extend(data['labels'])

    n_pos = sum(1 for l in all_labels if l == 1)
    n_neg = sum(1 for l in all_labels if l == 0)
    n_ignore = sum(1 for l in all_labels if l == -1)

    print(f"\nTraining data statistics:")
    print(f"  Total proposals: {len(all_labels)}")
    print(f"  Positive (pothole): {n_pos} ({100*n_pos/len(all_labels):.1f}%)")
    print(f"  Negative (background): {n_neg} ({100*n_neg/len(all_labels):.1f}%)")
    print(f"  Ignored: {n_ignore} ({100*n_ignore/len(all_labels):.1f}%)")

    return training_data


def get_best_proposals(proposals, gt_boxes, top_k=10):
    """
    Get the top-k proposals with highest IoU for each GT box.

    Args:
        proposals: list of proposals
        gt_boxes: list of ground truth boxes
        top_k: number of best proposals per GT

    Returns:
        best_proposals: list of (proposal_idx, gt_idx, iou) tuples
    """
    if len(proposals) == 0 or len(gt_boxes) == 0:
        return []

    iou_matrix = compute_iou_matrix(proposals, gt_boxes)

    best_proposals = []
    for gt_idx in range(len(gt_boxes)):
        ious = iou_matrix[:, gt_idx]
        top_indices = np.argsort(ious)[::-1][:top_k]

        for prop_idx in top_indices:
            if ious[prop_idx] > 0:
                best_proposals.append((prop_idx, gt_idx, ious[prop_idx]))

    return best_proposals


if __name__ == "__main__":
    # Test IoU computation
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]  # 50% overlap

    iou = compute_iou(box1, box2)
    print(f"IoU between overlapping boxes: {iou:.3f}")

    # Test with no overlap
    box3 = [20, 20, 30, 30]
    iou = compute_iou(box1, box3)
    print(f"IoU between non-overlapping boxes: {iou:.3f}")

    # Test label assignment
    proposals = [
        [0, 0, 10, 10],   # Perfect match with GT
        [5, 5, 15, 15],   # Partial overlap
        [50, 50, 60, 60], # No overlap
    ]
    gt_boxes = [[0, 0, 10, 10]]

    labels, ious, matched = assign_labels_to_proposals(proposals, gt_boxes)
    print(f"\nLabel assignment test:")
    for i, (label, iou) in enumerate(zip(labels, ious)):
        status = {1: 'positive', 0: 'negative', -1: 'ignore'}[label]
        print(f"  Proposal {i}: IoU={iou:.2f}, label={status}")
