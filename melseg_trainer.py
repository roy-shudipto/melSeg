"""
This pipeline trains a semantic segmentation model.

Example usage:
python3 melseg_trainer.py --config example_configs/training_config_01.yaml --cuda_id 0
"""
import click
import pathlib
from loguru import logger

from melseg.config import parse_config, TrainingConfig
from melseg.model import (
    get_model,
    get_device,
    get_optimizer,
    get_loss_function,
)
from melseg.dataset import SegDataset
from melseg.trainer import Trainer


def train(config: dict, config_ref=None, cuda_id=None) -> None:
    # get training-config
    training_config = TrainingConfig(config)
    logger.info("Successfully read training-config.")

    # create a directory for checkpoint
    training_config.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Successfully created checkpoint directory:"
        f" {training_config.checkpoint_directory}"
    )

    # save training-config in checkpoint-directory
    training_config.save()
    logger.info(f"Training-config is saved as: {training_config.config_dst}")

    # initiate segmentation dataset
    seg_dataset = SegDataset(
        training_config.dataset_root, training_config.cross_validation_fold
    )

    # run training for each fold
    for fold in range(training_config.cross_validation_fold):
        # get training-fold reference
        fold_ref = f"Fold: {fold + 1}/{training_config.cross_validation_fold}"

        logger.info(f"Starting training for {fold_ref}")

        # define model
        model = get_model()
        device = get_device(cuda_id=cuda_id)
        loss_func = get_loss_function(training_config.loss_func)
        optimizer = get_optimizer(
            training_config.optimizer,
            model,
            training_config.learning_rate,
        )
        logger.info(f"Model is initialized to get trained on: {device}")

        # get dataloader
        dataloader_dict = seg_dataset.get_dataloader(fold, training_config.batch_size)

        # get training reference
        training_ref = fold_ref if not config_ref else f"{config_ref} | {fold_ref}"

        # get trainer
        trainer = Trainer(
            model=model,
            device=device,
            loss_func=loss_func,
            optimizer=optimizer,
            epochs=training_config.epochs,
            train_dataloader=dataloader_dict["train"],
            val_dataloader=dataloader_dict["val"],
            checkpoint_path=training_config.get_checkpoint_path(fold + 1),
            log_path=training_config.get_log_path(fold + 1),
            write_checkpoint=training_config.write_checkpoint,
            training_ref=training_ref,
        )

        # run trainer
        trainer.run()

    logger.info("Successfully completed the training.")


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="Path to the training-config [.yaml] file.",
)
@click.option(
    "--cuda_id",
    type=int,
    required=False,
    help="CUDA-ID to train on a specific CUDA-GPU.",
)
def run_training(config, cuda_id) -> None:
    # convert config from str to pathlib.Path
    config_path = pathlib.Path(config)

    # multiple training-configs will be generated if parameters contain a list
    # of values instead of a single value.
    config_variations = parse_config(config_path)
    logger.info(f"Number of training-config variations: {len(config_variations)}")

    # train
    for idx, config_variation in enumerate(config_variations):
        # get reference for training-config
        config_ref = f"Config: {idx + 1}/{len(config_variations)}"

        # train
        logger.info(f"Training with {config_ref}.")
        train(config_variation, config_ref, cuda_id)


if __name__ == "__main__":
    logger.info("This pipeline trains a semantic segmentation model.")
    run_training()
