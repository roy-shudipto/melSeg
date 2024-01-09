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
        write_checkpoint,
    ) -> None:
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.checkpoint_path = checkpoint_path
        self.write_checkpoint = write_checkpoint

    def run(self) -> None:
        self.model.to(self.device)

        for epoch in range(self.epochs):
            epoch_losses = {"train": [], "val": []}

            for phase in ["train", "val"]:

                if phase == "train":
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                logger.info(f"\nRunning => Epoch {epoch} | Phase: {phase}")
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

            logger.info(
                f"Epoch: {epoch}; Train Loss: {mean(epoch_losses['train'])}; Val Loss: {mean(epoch_losses['val'])}"
            )

        if self.write_checkpoint:
            torch.save(self.model.state_dict(), self.checkpoint_path)
