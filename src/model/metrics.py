from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassStatScores

__all__ = ["MyAwesomeMetricsWrapper"]


class MyAwesomeMetricsWrapper:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=num_classes,
            normalize="true",
            ignore_index=0,
        )
        self.stat_scores = MulticlassStatScores(
            num_classes=num_classes,
            average="none",
            ignore_index=0,
        )
        self.awesome_metric = MyAwesomeMetric(num_classes=num_classes)

        self.metrics_per_split = {
            "train": [self.stat_scores],
            "val": [self.stat_scores],
            "test": [self.stat_scores, self.confusion_matrix, self.awesome_metric],
        }

    def update(self, preds, target, split):
        for metric in self.metrics_per_split[split]:
            metric.update(preds, target)

    def reset(self, split):
        for metric in self.metrics_per_split[split]:
            metric.reset()

    def compute(self, split) -> dict:
        metrics = {}
        if metric in self.metrics_per_split[split]:
            # Define custom post-processing per metric
            if metric == self.confusion_matrix:
                confmat = self.confusion_matrix.compute()
                fig, ax = confmat.plot()
                metrics["ConfusionMatrix"] = fig
            elif metric == self.stat_scores:
                tp, fp, tn, fn, sup = self.stat_scores.compute()
                metrics["Precision"] = tp / (tp + fp)
                metrics["Recall"] = tp / (tp + fn)
            else:
                metrics[metric.__name__] = metric.compute()
        return metrics


class MyAwesomeMetric(Metric):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def update(self, preds, target):
        pass

    def compute(self):
        pass
