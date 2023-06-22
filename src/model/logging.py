from .utils import *

__all__ = ["SummaryImg"]


class SummaryImg:
    def __init__(self, n_classes, logger):
        self.n_classes = n_classes
        self.logger = logger

    def __call__(self, x, y, y_hat):
        raise NotImplementedError("Implement me!")

    def train_image(self, x, y, y_hat):
        # TODO: Implement training image processing and logging here
        # grid = make_grid(imgs, nrow=4, padding=6, pad_value=1)
        # self.logger.log_image(f"{stage}_images", [to_pil_image(grid)])
        raise NotImplementedError("Implement me!")

    def val_image(self, x, y, y_hat):
        # TODO: Implement validation image processing and logging here
        raise NotImplementedError("Implement me!")

    def test_image(self, x, y, y_hat):
        # TODO: Implement test image processing and logging here
        raise NotImplementedError("Implement me!")
