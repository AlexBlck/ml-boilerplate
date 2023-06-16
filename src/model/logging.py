from .utils import *

__all__ = ["SummaryImg"]


class SummaryImg:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, x, y, y_hat):
        raise NotImplementedError("Implement me!")
