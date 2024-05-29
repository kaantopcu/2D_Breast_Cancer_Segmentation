"""# Performance Metrics"""

import numpy as np
from scipy.spatial.distance import cdist,directed_hausdorff

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    if total_pixel_pred == 0:
      return 0.0  # Avoid division by zero
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3)

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    if total_pixel_truth == 0:
        return 0.0  # Avoid division by zero
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3)

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    if total_sum == 0:
        return 0.0  # Avoid division by zero
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places


def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    if union == 0:
        return 0.0  # Avoid division by zero
    iou = np.mean(intersect/union)
    return round(iou, 3)



def hausdorff_distance(true_segmentation, predicted_segmentation):
    true_surface = extract_surface(true_segmentation)
    predicted_surface = extract_surface(predicted_segmentation)

    if len(true_surface) == 0 or len(predicted_surface) == 0:
        return float('inf')  # Avoid division by zero

    hd_forward = directed_hausdorff(true_surface, predicted_surface)[0]
    hd_backward = directed_hausdorff(predicted_surface, true_surface)[0]

    hd = max(hd_forward, hd_backward)
    return round(hd, 3)


def average_symmetric_surface_distance(true_segmentation, predicted_segmentation):
    true_surface = extract_surface(true_segmentation)
    predicted_surface = extract_surface(predicted_segmentation)

    if len(true_surface) == 0 or len(predicted_surface) == 0:
        return float('inf')  # Avoid division by zero

    distances_true_to_pred = cdist(true_surface, predicted_surface, metric='euclidean')
    min_distances_true_to_pred = np.min(distances_true_to_pred, axis=1)

    distances_pred_to_true = cdist(predicted_surface, true_surface, metric='euclidean')
    min_distances_pred_to_true = np.min(distances_pred_to_true, axis=1)

    assd = (np.mean(min_distances_true_to_pred) + np.mean(min_distances_pred_to_true)) / 2.0
    return round(assd, 3)

def volumetric_similarity(groundtruth_mask, pred_mask):
    volume_true = np.sum(groundtruth_mask)
    volume_pred = np.sum(pred_mask)

    if volume_true + volume_pred == 0:
        return float('inf')  # Avoid division by zero

    vs = 1 - abs(volume_true - volume_pred) / (volume_true + volume_pred)
    return round(vs, 3)

def specificity(groundtruth_mask, pred_mask):
    true_negatives = np.sum((groundtruth_mask == 0) & (pred_mask == 0))
    false_positives = np.sum((groundtruth_mask == 0) & (pred_mask == 1))
    if true_negatives + false_positives == 0:
        return float('inf')  # Avoid division by zero
    specificity = true_negatives / (true_negatives + false_positives)
    return round(specificity, 3)


def matthews_corrcoef(groundtruth_mask, pred_mask):
    TP = np.sum((groundtruth_mask == 1) & (pred_mask == 1))
    TN = np.sum((groundtruth_mask == 0) & (pred_mask == 0))
    FP = np.sum((groundtruth_mask == 0) & (pred_mask == 1))
    FN = np.sum((groundtruth_mask == 1) & (pred_mask == 0))

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        return float('inf')  # Avoid division by zero
    mcc = numerator / denominator
    return round(mcc, 3)

def extract_surface(segmentation):
    """ Extracts the surface points from a binary segmentation mask. """
    # Assuming segmentation is a binary mask where 1 is the structure and 0 is the background.
    surface_points = np.argwhere(segmentation)
    return surface_points

def calculate_diameter(points):
    """ Calculates the diameter of the set of points. """
    if len(points) == 0:
        return 0
    distances = cdist(points, points)
    diameter = np.max(distances)
    return diameter

def normalized_surface_distance(true_segmentation, predicted_segmentation):
    """ Calculates the Normalized Surface Distance (NSD) between two segmentations. """
    true_surface = extract_surface(true_segmentation)
    predicted_surface = extract_surface(predicted_segmentation)

    if len(true_surface) == 0 or len(predicted_surface) == 0:
        return float('inf')  # Avoid division by zero

    distances = cdist(true_surface, predicted_surface, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    diameter = calculate_diameter(true_surface)
    if diameter == 0:
        return float('inf')  # Avoid division by zero

    nsd = np.mean(min_distances / diameter)
    return nsd

def compute_average_metrics(groundtruth_masks, pred_masks):
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    dice_scores = []
    iou_scores = []
    hausdorff_scores = []
    assd_scores = []
    vs_scores = []
    specificity_scores = []
    mcc_socres = []
    nsd_scores = []

    for groundtruth_mask, pred_mask in zip(groundtruth_masks, pred_masks):
        precision_scores.append(precision_score_(groundtruth_mask, pred_mask))
        recall_scores.append(recall_score_(groundtruth_mask, pred_mask))
        accuracy_scores.append(accuracy(groundtruth_mask, pred_mask))
        dice_scores.append(dice_coef(groundtruth_mask, pred_mask))
        iou_scores.append(iou(groundtruth_mask, pred_mask))
        hausdorff_scores.append(hausdorff_distance(groundtruth_mask, pred_mask))
        assd_scores.append(average_symmetric_surface_distance(groundtruth_mask, pred_mask))
        vs_scores.append(volumetric_similarity(groundtruth_mask, pred_mask))
        specificity_scores.append(specificity(groundtruth_mask, pred_mask))
        mcc_socres.append(matthews_corrcoef(groundtruth_mask, pred_mask))
        nsd_scores.append(normalized_surface_distance(groundtruth_mask, pred_mask))



    average_metrics = {
        'Precision': round(np.mean(precision_scores), 3),
        'Recall': round(np.mean(recall_scores), 3),
        'Accuracy': round(np.mean(accuracy_scores), 3),
        'Dice Coefficient': round(np.mean(dice_scores), 3),
        'Jaccard Index': round(np.mean(iou_scores), 3),
        'Hausdorff Distance': round(np.mean(hausdorff_scores), 3),
        'Average Symmetric Surface Distance': round(np.mean(assd_scores), 3),
        'Volumetric Similarity (VS)': round(np.mean(vs_scores), 3),
        'Specificity (True Negative Rate)': round(np.mean(specificity_scores), 3),
        'Matthews Correlation Coefficient': round(np.mean(mcc_socres), 3),
        'Normalized Surface Distance': round(np.mean(nsd_scores), 3)

    }

    return average_metrics