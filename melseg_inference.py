"""
This pipeline runs inference using trained melSeg models.

Example usage:
python3 melseg_inference.py --help
"""

import click
import numpy as np
import pathlib
import torch
from loguru import logger
from PIL import Image
from transformers import SamProcessor

from melseg.model import (
    load_checkpoint,
    get_device,
)
from melseg.sam_dataset import get_bounding_box
from melseg.defaults import SAM_INPUT_SIZE, SAM_PRETRAINED_MODEL


def preprocess_input(*, image_path: str, mask_path: str, device: torch.device) -> dict:
    # load image
    try:
        image = Image.open(image_path)
    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"Unable to load image from: {image_path}")
        logger.error(f"Error: {e}")
        exit(1)

    # load mask
    try:
        ground_truth_mask = Image.open(mask_path)
    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"Unable to load mask from: {mask_path}")
        logger.error(f"Error: {e}")
        exit(1)

    # resize to match input-size for SAM
    image = image.resize(SAM_INPUT_SIZE)
    ground_truth_mask = ground_truth_mask.resize(SAM_INPUT_SIZE)

    # conver PIL Image to np.array
    ground_truth_mask = np.array(ground_truth_mask)
    ground_truth_mask = (ground_truth_mask / 255.0).astype(np.uint8)

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    processor = SamProcessor.from_pretrained(SAM_PRETRAINED_MODEL)
    input = processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # move the input tensor to the device and return
    return {k: v.to(device) for k, v in input.items()}


def postprocess_output(output) -> Image:
    # apply sigmoid
    output = torch.sigmoid(output.pred_masks.squeeze(1))

    # convert soft mask to hard mask
    output = output.cpu().numpy().squeeze()
    output = (output > 0.5).astype(np.uint8)

    # convert numpy array to Pillow->Image and return
    return Image.fromarray(np.uint8(output * 255))


@click.command()
@click.option(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a meSeg checkpoint in [.pt] format.",
)
@click.option(
    "--image",
    type=str,
    required=True,
    help="Path to an image for running segmentation.",
)
@click.option(
    "--mask",
    type=str,
    required=True,
    help="Path to the mask of given image for running segmentation.",
)
@click.option(
    "--save",
    type=str,
    required=True,
    help="Path to save the segmentation result in [.png] format.",
)
def run_inference(checkpoint, image, mask, save) -> None:
    # load checkpoint's state-dict
    model = load_checkpoint(pathlib.Path(checkpoint))
    logger.info(f">> Loaded checkpoint from: {checkpoint}")

    # get device
    device = get_device()
    logger.info(f">> Available device: {device}")

    # send model to device
    model.to(device)
    logger.info(f">> Model is sent to device: {device}")

    # set model to evaluate mode
    model.eval()
    logger.info(">> Model is set to eval-mode.")

    # preprocess input for inference
    input = preprocess_input(image_path=image, mask_path=mask, device=device)
    logger.info(">> Pre-processed input.")

    # run model to get prediction
    with torch.no_grad():
        output = model(**input, multimask_output=False)
        prediction = postprocess_output(output)
        logger.info(">> Model has completed prediction.")

    # save prediction
    prediction.save(save)
    logger.info(f">> Prediction is saved as: {save}")


if __name__ == "__main__":
    logger.info("This pipeline runs inference using trained melSeg models.")
    run_inference()
