import numpy as np
from loguru import logger
from statistics import mean


class Metric:
    def __init__(self) -> None:
        self.loss = {"train": None, "val": None}
        self.accuracy = {"train": None, "val": None}
        self.precision = {"train": None, "val": None}
        self.recall = {"train": None, "val": None}
        self.dice = {"train": None, "val": None}
        self.iou = {"train": None, "val": None}


class PerformanceMetric:
    def __init__(self) -> None:
        self.loss_list = {"train": [], "val": []}
        self.accuracy_list = {"train": [], "val": []}
        self.precision_list = {"train": [], "val": []}
        self.recall_list = {"train": [], "val": []}
        self.dice_list = {"train": [], "val": []}
        self.iou_list = {"train": [], "val": []}

    @classmethod
    def calc_accuracy(cls, groundtruth_mask, pred_mask):
        intersect = np.sum(pred_mask * groundtruth_mask)
        union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
        xor = np.sum(groundtruth_mask == pred_mask)
        acc = np.mean(xor / (union + xor - intersect))
        return acc

    @classmethod
    def calc_precision(cls, groundtruth_mask, pred_mask):
        intersect = np.sum(pred_mask * groundtruth_mask)
        total_pixel_pred = np.sum(pred_mask)
        precision = np.mean(intersect / total_pixel_pred)
        return precision

    @classmethod
    def calc_recall(cls, groundtruth_mask, pred_mask):
        intersect = np.sum(pred_mask * groundtruth_mask)
        total_pixel_truth = np.sum(groundtruth_mask)
        recall = np.mean(intersect / total_pixel_truth)
        return recall

    @classmethod
    def calc_dice(cls, groundtruth_mask, pred_mask):
        intersect = np.sum(pred_mask * groundtruth_mask)
        total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
        dice = np.mean(2 * intersect / total_sum)
        return dice

    @classmethod
    def calc_iou(cls, groundtruth_mask, pred_mask):
        intersect = np.sum(pred_mask * groundtruth_mask)
        union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
        iou = np.mean(intersect / union)
        return iou

    def run_metric_calculation(self, groundtruth_mask, prediction_mask):
        return {
            "accuracy": self.calc_accuracy(groundtruth_mask, prediction_mask),
            "precision": self.calc_precision(groundtruth_mask, prediction_mask),
            "recall": self.calc_recall(groundtruth_mask, prediction_mask),
            "dice": self.calc_dice(groundtruth_mask, prediction_mask),
            "iou": self.calc_iou(groundtruth_mask, prediction_mask),
        }

    def update(self, *, phase, loss, groundtruth_mask, prediction_mask):
        # check: groundtruth_mask
        groundtruth_mask = np.array(groundtruth_mask)
        if len(groundtruth_mask.shape) != 1:
            logger.error(
                "For metric calculaition, groundtruth-mask is expected to be a 1-d array."
            )
            exit(1)

        # check: prediction_mask
        prediction_mask = np.array(prediction_mask)
        if len(prediction_mask.shape) != 1:
            logger.error(
                "For metric calculaition, prediction-mask is expected to be a 1-d array."
            )
            exit(1)

        # calculate metric values
        metric_dict = self.run_metric_calculation(groundtruth_mask, prediction_mask)

        # update metric: loss
        self.loss_list[phase].append(loss)

        # update metric: accuracy
        self.accuracy_list[phase].append(metric_dict["accuracy"])

        # update metric: precision
        self.precision_list[phase].append(metric_dict["precision"])

        # update metric: recall
        self.recall_list[phase].append(metric_dict["recall"])

        # update metric: dice
        self.dice_list[phase].append(metric_dict["dice"])

        # update metric: iou
        self.iou_list[phase].append(metric_dict["iou"])

    def get_metric(self):
        metric = Metric()

        for phase in ["train", "val"]:
            metric.loss[phase] = round(mean(self.loss_list[phase]), 4)
            metric.accuracy[phase] = round(mean(self.accuracy_list[phase]), 4)
            metric.precision[phase] = round(mean(self.precision_list[phase]), 4)
            metric.recall[phase] = round(mean(self.recall_list[phase]), 4)
            metric.dice[phase] = round(mean(self.dice_list[phase]), 4)
            metric.iou[phase] = round(mean(self.iou_list[phase]), 4)

        return metric
