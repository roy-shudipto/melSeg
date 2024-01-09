import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import SamProcessor

from melseg.defaults import SAM_INPUT_SIZE, SAM_PRETRAINED_MODEL


# get bounding boxes from mask
def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


# SAM Dataset
class SAMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.processor = SamProcessor.from_pretrained(SAM_PRETRAINED_MODEL)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # read image and mask
        item = self.dataset[idx]
        image = Image.open(item["image"])
        ground_truth_mask = Image.open(item["mask"])

        # resize to match input-size for SAM
        image = image.resize(SAM_INPUT_SIZE)
        ground_truth_mask = ground_truth_mask.resize(SAM_INPUT_SIZE)

        # conver PIL Image to np.array
        ground_truth_mask = np.array(ground_truth_mask)
        ground_truth_mask = (ground_truth_mask / 255.0).astype(np.uint8)

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs
