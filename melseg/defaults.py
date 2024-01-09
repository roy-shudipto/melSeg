CHECKPOINT_EXTENSION = ".pt"
CONFIG_EXTENSION = ".yaml"
CONFIG_OUTPUT_NAME = "training_config.yaml"
LOG_EXTENSION = ".csv"
LOG_HEADERS = [
    "EPOCH",
    "TRAIN LOSS",
]
LOSS_FUNC_LIST = [
    "DiceCELoss",
    "FocalLoss",
]
OPTIMIZER_LIST = ["ADAM", "SGD"]

# SAM
SAM_PRETRAINED_MODEL = "facebook/sam-vit-base"
SAM_INPUT_SIZE = (256, 256)


# HAM10000 Mask naming convention
HAM10000_MASK_NAMING_CONVENTION = "_segmentation"
HAM10000_MASK_EXTENSION = ".png"
