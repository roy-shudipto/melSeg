import pandas as pd

from melseg.defaults import LOG_HEADERS


class TrainingLog:
    def __init__(self):
        self.log = pd.DataFrame(columns=LOG_HEADERS)

    def update(self, *, epoch, metric):
        df = pd.DataFrame(
            {
                "EPOCH": [epoch],
                "TRAIN LOSS": [metric.loss["train"]],
                "TRAIN ACCURACY": [metric.accuracy["train"]],
                "TRAIN PRECISION": [metric.precision["train"]],
                "TRAIN RECALL": [metric.recall["train"]],
                "TRAIN DICE": [metric.dice["train"]],
                "TRAIN IOU": [metric.iou["train"]],
                "VAL LOSS": [metric.loss["val"]],
                "VAL ACCURACY": [metric.accuracy["val"]],
                "VAL PRECISION": [metric.precision["val"]],
                "VAL RECALL": [metric.recall["val"]],
                "VAL DICE": [metric.dice["val"]],
                "VAL IOU": [metric.iou["val"]],
            }
        )

        self.log = df if self.log.empty is True else pd.concat([self.log, df], axis=0)

    def save(self, path):
        self.log.to_csv(path, index=False)
