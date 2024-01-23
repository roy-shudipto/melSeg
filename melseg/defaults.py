ANALYSIS_LOG_NAME = "log_analysis.csv"
CHECKPOINT_EXTENSION = ".pt"
CONFIG_EXTENSION = ".yaml"
CONFIG_NAME = "training_config.yaml"
DATASET_SHUFFLE_SEED = 5
LOG_EXTENSION = ".csv"
LOG_HEADERS = [
    "EPOCH",
    "TRAIN LOSS",
    "TRAIN ACCURACY",
    "TRAIN PRECISION",
    "TRAIN RECALL",
    "TRAIN DICE",
    "TRAIN IOU",
    "VAL LOSS",
    "VAL ACCURACY",
    "VAL PRECISION",
    "VAL RECALL",
    "VAL DICE",
    "VAL IOU",
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
