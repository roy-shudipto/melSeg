import click
import cv2
import numpy as np
import pathlib
from loguru import logger
from typing import Any


def get_contours(image: np.ndarray) -> Any:
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding or Canny edge detection
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


@click.command()
@click.option(
    "--image_tag",
    type=str,
    required=True,
    help="Image-file tag. Eg: ISIC_0024306",
)
@click.option(
    "--result_root",
    type=str,
    required=True,
    help="Path to the result-directory.",
)
def run(image_tag: str, result_root: str) -> None:
    # generate paths
    image_input_path = pathlib.Path(result_root) / pathlib.Path(image_tag).with_suffix(
        ".jpg"
    )
    image_prediction_path = pathlib.Path(result_root) / pathlib.Path(
        f"{image_tag}_prediction"
    ).with_suffix(".png")
    image_mask_path = pathlib.Path(result_root) / pathlib.Path(
        f"{image_tag}_segmentation"
    ).with_suffix(".png")
    result_viz_path = pathlib.Path(result_root) / pathlib.Path(
        f"{image_tag}_result_viz"
    ).with_suffix(".jpg")

    # try to read images
    try:
        image_input = cv2.imread(image_input_path.as_posix(), 1)
        image_prediction = cv2.imread(image_prediction_path.as_posix(), 1)
        image_mask = cv2.imread(image_mask_path.as_posix(), 1)
    except (FileNotFoundError, RuntimeError):
        logger.error(
            f"Failed to read all required files using image_tag: [{image_tag}] from"
            f" {result_root}"
        )
        exit(1)

    # make a copy of image-input
    image_draw = image_input.copy()

    # draw all contours of prediction on the image in yellow
    cv2.drawContours(image_draw, get_contours(image_prediction), -1, (0, 255, 255), 2)

    # draw all contours of ground-truth mash on the image in green
    cv2.drawContours(image_draw, get_contours(image_mask), -1, (0, 255, 0), 2)

    # display the input-image with both contours
    cv2.imshow("Segmentation Result Visualization", image_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save result
    cv2.imwrite(result_viz_path.as_posix(), image_draw)

    return None


if __name__ == "__main__":
    logger.info(
        "This pipeline draws the contours of prediction and ground-truth mask on the"
        " image for visualization."
    )
    run()
