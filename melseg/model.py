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
def get_device(cuda_id=None) -> torch.device:
    # check: cuda is available
    if torch.cuda.is_available() is False:
        logger.debug("CUDA is not available.")
        return torch.device("cpu")

    # default cuda is cuda:0
    if not cuda_id:
        return torch.device("cuda:0")

    # select cuda-id
    cuda_ids = [i for i in range(torch.cuda.device_count())]

    if cuda_id in cuda_ids:
        return torch.device(f"cuda:{cuda_id}")
    else:
        logger.error(f"[CUDA-ID: {cuda_id}] is not supported.")
        logger.error(f"Available CUDA-IDs are: {cuda_ids}.")
        exit(1)


# loss-function
def get_loss_function(loss_func_name):
    # define loss-function
    if loss_func_name == "DiceCELoss":
        return monai.losses.DiceCELoss(sigmoid=True, squared_pred=True)
    elif loss_func_name == "FocalLoss":
        monai.losses.FocalLoss()
        return monai.losses.FocalLoss()
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
    if optimizer_name.upper() == "SGD":
        return optim.SGD(
            model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        logger.error(f"[OPTIMIZER: {optimizer_name}] is not supported by PyTorch.")
        exit(1)
