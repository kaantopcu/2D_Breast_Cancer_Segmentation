import torch
from transformers import SamModel, SamProcessor
from utils.bounding_box import get_bounding_box
import numpy as np

def load_model_and_processor(model_name, processor_name, device):
    model = SamModel.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(processor_name)
    return model, processor

def inference(image, processor, model, device):
    ground_truth_seg = np.array(image["label"])
    input_boxes = get_bounding_box(ground_truth_seg)
    inputs = processor(image["image"], input_boxes=[[input_boxes]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    seg_prob = seg_prob.cpu().numpy().squeeze()
    seg = (seg_prob > 0.5).astype(np.uint8)
    return seg
