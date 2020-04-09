import os

from PIL import Image
from torch.utils.data import Dataset


class WildHandsDataset(Dataset):
    """
    Args:
        dataset_path (string): folder of the extracted dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
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
            tuple: (image, target) where target is the same image.
        """
        pass

# Testing
if __name__ == "__main__":
    pass