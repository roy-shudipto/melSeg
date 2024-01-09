import torch
from loguru import logger
from statistics import mean
from tqdm import tqdm


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

    def run(self) -> None:
        self.model.to(self.device)

        # variables to find the best-checkpoint (in validation)
        best_loss = None
        best_epoch = None
        best_model_state_dict = {}
        for epoch in range(self.epochs):
            epoch_losses = {"train": [], "val": []}

            for phase in ["train", "val"]:

                if phase == "train":
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                logger.info(f"Running => Epoch: {epoch} | Phase: {phase}")
                for batch in tqdm(self.dataloaders[phase]):
                    # forward pass
                    outputs = self.model(
                        pixel_values=batch["pixel_values"].to(self.device),
                        input_boxes=batch["input_boxes"].to(self.device),
                        multimask_output=False,
                    )

                    # compute loss
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = (
                        batch["ground_truth_mask"].float().to(self.device)
                    )
                    loss = self.loss_func(
                        predicted_masks, ground_truth_masks.unsqueeze(1)
                    )

                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    epoch_losses[phase].append(loss.item())

            # calc. mean loss
            train_loss = mean(epoch_losses["train"])
            val_loss = mean(epoch_losses["val"])
            logger.info(
                f"Epoch: {epoch}; Train Loss: {train_loss}; Val Loss: {val_loss}"
            )

            # evaluate performance for this epoch
            if best_loss is None or val_loss <= best_loss:
                best_loss = val_loss
                best_epoch = epoch
                for k, v in self.model.state_dict().items():
                    best_model_state_dict[k] = v.cpu()

                logger.info("Better performance is FOUND.")
                logger.info(f"Best epoch: {best_epoch}; Best Val Loss: {best_loss};")
            else:
                logger.info("Better performance is NOT FOUND.")
                logger.info(f"Best epoch: {best_epoch}; Best Val Loss: {best_loss};")

        # save the best-checkpoint (in validation)
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

        # # save training-log
        # training_log.save(path=self.log_path)
        # logger.info(f"Training-log is saved as: {self.log_path}")
