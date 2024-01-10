import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


from melseg.log import TrainingLog
from melseg.metric import PerformanceMetric


class Trainer:
    def __init__(
        self,
        *,
        model,
        device,
        loss_func,
        optimizer,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_path,
        log_path,
        write_checkpoint,
        training_ref="",
    ) -> None:
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.write_checkpoint = write_checkpoint
        self.training_ref = training_ref

    def run(self) -> None:
        # initiate training-logging
        training_log = TrainingLog()

        # send model to device
        self.model.to(self.device)

        # variables to find the best-checkpoint (in validation)
        best_loss = None
        best_epoch = None
        best_model_state_dict = {}

        # run training for each epoch
        for epoch in range(self.epochs):
            # initiate performance metric
            pm = PerformanceMetric()

            # run training for each phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                logger.info(
                    f"Running >> {self.training_ref} | Epoch: {epoch} | Phase: {phase}"
                )
                for batch in tqdm(self.dataloaders[phase]):
                    # forward pass
                    outputs = self.model(
                        pixel_values=batch["pixel_values"].to(self.device),
                        input_boxes=batch["input_boxes"].to(self.device),
                        multimask_output=False,
                    )

                    # compute loss
                    ground_truth_masks = (
                        batch["ground_truth_mask"].float().to(self.device)
                    ).unsqueeze(
                        1
                    )  # [batch, 1, 256, 256]
                    predicted_masks = outputs.pred_masks.squeeze(
                        1
                    )  # [batch, 1, 256, 256]
                    loss = self.loss_func(predicted_masks, ground_truth_masks)

                    # backward pass
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # update performance metric
                    gt = (
                        ground_truth_masks.clone()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                        .flatten()
                    )
                    pred = predicted_masks.clone().detach()
                    pred = torch.sigmoid(pred)
                    pred = pred.cpu().numpy().squeeze()
                    pred = (pred > 0.5).astype(np.uint8).flatten()

                    pm.update(
                        phase=phase,
                        loss=loss.item(),
                        groundtruth_mask=gt,
                        prediction_mask=pred,
                    )

            # compute performance metric for this epoch
            metric = pm.get_metric()
            train_loss = metric.loss["train"]
            val_loss = metric.loss["val"]
            logger.info(
                f"Epoch: {epoch}; Train Loss: {train_loss}; Val Loss: {val_loss}"
            )

            # update training-log
            training_log.update(epoch=epoch, metric=metric)

            # evaluate training-performance for this epoch
            if best_loss is None or val_loss <= best_loss:
                best_loss = val_loss
                best_epoch = epoch
                for k, v in self.model.state_dict().items():
                    best_model_state_dict[k] = v.cpu()

                logger.info("Better performance is ACHIEVED.")
                logger.info(f"Best Epoch: {best_epoch}; Best Val Loss: {best_loss};")
            else:
                logger.info("Better performance is NOT ACHIEVED.")
                logger.info(f"Best Epoch: {best_epoch}; Best Val Loss: {best_loss};")

        # save the overall best-checkpoint (in validation)
        if self.write_checkpoint:
            torch.save(
                {
                    "loss": best_loss,
                    "epoch": best_epoch,
                    "model_state_dict": best_model_state_dict,
                },
                self.checkpoint_path,
            )
            logger.info(f"Checkpoint is saved as: {self.checkpoint_path}")

        # save training-log
        training_log.save(path=self.log_path)
        logger.info(f"Training-log is saved as: {self.log_path}")
