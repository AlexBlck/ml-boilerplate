import torch
from torchvision import transforms

__all__ = ["MyAwesomeTransforms"]


class MyAwesomeTransforms:
    def __init__(self, size):
        self.transform = {"train": {}, "val": {}, "test": {}, "predict": {}}
        # Image transforms
        image_no_augmentations = transforms.Compose(
            [
                lambda x: x.type(torch.uint8),
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_augmentations = transforms.Compose(
            [
                lambda x: x.type(torch.uint8),
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform["train"]["image"] = image_augmentations
        for split in ["val", "test", "predict"]:
            self.transform[split]["image"] = image_no_augmentations

        # Mask transforms
        resize_mask = transforms.Compose(
            [
                transforms.Resize((size, size)),
            ]
        )
        for split in ["train", "val", "test", "predict"]:
            self.transform[split]["mask"] = resize_mask

    def __call__(self, sample, split):
        for k, v in sample.items():
            sample[k] = self.transform[split][k](v)
        return sample
