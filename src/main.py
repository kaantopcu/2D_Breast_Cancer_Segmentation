import torch
from data.load_data import load_sample_image
from inference.model_inference import load_model_and_processor, inference
from metrics.metrics import compute_average_metrics, sample_image_metrics
from visualization.visualize import show_box, show_boxes_on_image
from utils.bounding_box import get_bounding_box
import numpy as np
from metrics.metrics import precision_score_,recall_score_,accuracy,dice_coef,iou,hausdorff_distance,average_symmetric_surface_distance,volumetric_similarity,specificity,matthews_corrcoef,normalized_surface_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and processors
sam_model, sam_processor = load_model_and_processor("facebook/sam-vit-huge", "facebook/sam-vit-huge", device)
medsam_model, medsam_processor = load_model_and_processor("wanglab/medsam-vit-base", "wanglab/medsam-vit-base", device)

# Load sample data
dataset_name = "nielsr/breast-cancer"
split = "train"
idx = 10
sample_image, sample_ground_truth_seg = load_sample_image(dataset_name, split, idx)

# Show sample image with bounding boxes
input_boxes = get_bounding_box(sample_ground_truth_seg)
show_boxes_on_image(sample_image["image"], [input_boxes])

# Perform sample image inference
medsam_seg_pred = inference(sample_image, medsam_processor, medsam_model, device)
sam_seg_pred = inference(sample_image, sam_processor, sam_model, device)

# Compute metrics for all images
groundtruth_masks = [np.array(load_sample_image(dataset_name, split, i)[1]) for i in range(130)]
medsam_seg_masks = [inference(load_sample_image(dataset_name, split, i)[0], medsam_processor, medsam_model, device) for i in range(130)]
sam_seg_masks = [inference(load_sample_image(dataset_name, split, i)[0], sam_processor, sam_model, device) for i in range(130)]

average_metrics_medsam = compute_average_metrics(groundtruth_masks, medsam_seg_masks)
average_metrics_sam = compute_average_metrics(groundtruth_masks, sam_seg_masks)

print("MedSAM Metrics: ", average_metrics_medsam)
print("SAM Metrics: ", average_metrics_sam)

sample_image_metrics(sample_ground_truth_seg,sam_seg_pred)

#print(compute_average_metrics(sample_ground_truth_seg, sam_seg_pred))
#print(compute_average_metrics(sample_ground_truth_seg, medsam_seg_pred))
