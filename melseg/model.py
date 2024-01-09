import monai
import torch
import torch.optim as optim
from loguru import logger
from transformers import SamModel

from melseg.defaults import SAM_PRETRAINED_MODEL


# model
def get_model() -> SamModel:
    # load a pretrained model
    model = SamModel.from_pretrained(SAM_PRETRAINED_MODEL)

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model


# device
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loss-function
def get_loss_function(loss_func_name):
    # define loss-function
    if loss_func_name == "DiceCELoss":
        return monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
    elif loss_func_name == "FocalLoss":
        return monai.losses.FocalLoss(sigmoid=True, squared_pred=True, reduction="mean")
    else:
        logger.error(f"[LOSS_FUNC: {loss_func_name}] is not supported.")
        exit(1)


# optimizer
def get_optimizer(optimizer_name, model, lr, weight_decay=0) -> optim.Optimizer:
    # define optimizer
    if optimizer_name.upper() == "ADAM":
        return optim.Adam(
            model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        logger.error(f"[OPTIMIZER: {optimizer_name}] is not supported by PyTorch.")
        exit(1)
