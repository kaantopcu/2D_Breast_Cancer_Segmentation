from datasets import load_dataset
import numpy as np

def load_sample_image(dataset_name, split, idx):
    dataset = load_dataset(dataset_name, split=split)
    sample_image = dataset[idx]
    sample_ground_truth_seg = np.array(sample_image["label"])
    return sample_image, sample_ground_truth_seg
