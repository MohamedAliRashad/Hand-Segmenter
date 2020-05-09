import os

from PIL import Image
from torch.utils.data import Dataset


class EGATEDataset(Dataset):
    """
    Extended GTEA Gaze+ Hand Dataset from <http://cbs.ic.gatech.edu/fpv/>

    Args:
        dataset_path: should include an image and its annotation.
        transform: list of pytorch transforms to apply on the image.
        target_transform: list of pytorch transforms to apply on the target.
    """

    def __init__(self, dataset_path, transform=None, target_transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is masks of the target classes.
        """
        pass


# Testing
if __name__ == "__main__":
    pass
