import numpy as np
import pathlib
from datasets import Dataset
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from melseg.defaults import (
    DATASET_SHUFFLE_SEED,
    HAM10000_MASK_EXTENSION,
    HAM10000_MASK_NAMING_CONVENTION,
)
from melseg.sam_dataset import SAMDataset


class SegDataset:
    def __init__(self, dataset_root: pathlib.Path, cross_validation_fold) -> None:
        self.dataset_root = dataset_root
        self.cross_validation_fold = cross_validation_fold
        self.skf = StratifiedKFold(n_splits=self.cross_validation_fold, shuffle=False)
        self.image_paths = []
        self.mask_paths = []

        self._init_dataset()

    def _init_dataset(self) -> None:
        # check: subdirectories ["images", "masks"] exist in the root.
        for dir_name in ["images", "masks"]:
            if (self.dataset_root / dir_name).exists:
                continue
            logger.error(
                f"Dataset directory:{self.dataset_root / dir_name} does not exist."
            )
            exit(1)

        # check: every image path has a mask path.
        for image_path in (self.dataset_root / "images").iterdir():
            mask_path = (
                self.dataset_root
                / "masks"
                / pathlib.Path(
                    image_path.stem + HAM10000_MASK_NAMING_CONVENTION
                ).with_suffix(HAM10000_MASK_EXTENSION)
            )

            if not mask_path.exists():
                logger.error(f"Image Path: {image_path}")
                logger.error(f"Mask Path: {mask_path} does not exist.")
                exit(1)

            self.image_paths.append(image_path.as_posix())
            self.mask_paths.append(mask_path.as_posix())

        # shuffle data
        self.image_paths, self.mask_paths = shuffle(
            self.image_paths, self.mask_paths, random_state=DATASET_SHUFFLE_SEED
        )

    def get_dataloader(self, fold_index, batch_size):
        # check: valid fold-index
        if fold_index < 0 or fold_index >= self.cross_validation_fold:
            logger.error(f"Number of folds: {self.cross_validation_fold}")
            logger.error(
                f"Fold index needs be within: 0 to {self.cross_validation_fold-1}"
            )
            logger.error(f"Received invalid fold index: {fold_index}")
            exit(1)

        # generate train and val datasets for the given fold-index
        for idx, (train_index, val_index) in enumerate(
            self.skf.split(self.image_paths, np.zeros(len(self.mask_paths)))
        ):
            if idx != fold_index:
                continue

            train_dataset = Dataset.from_dict(
                {
                    "image": [self.image_paths[i] for i in train_index],
                    "mask": [self.mask_paths[i] for i in train_index],
                }
            )

            val_dataset = Dataset.from_dict(
                {
                    "image": [self.image_paths[i] for i in val_index],
                    "mask": [self.mask_paths[i] for i in val_index],
                }
            )

        datasets = {"train": train_dataset, "val": val_dataset}
        dataloaders = {"train": None, "val": None}
        for phase in ["train", "val"]:
            # convert dataset into SAMDataset
            sam_dataset = SAMDataset(dataset=datasets[phase])

            # get dataloader of SAMDataset
            dataloaders[phase] = DataLoader(sam_dataset, batch_size=batch_size)

        return dataloaders
