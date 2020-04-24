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

    def __init__(self, dataset_path, level=1, transform=None, target_transform=None):
        self.dataset_path = os.path.join(dataset_path, f"level_{level}")
        self.transform = transform
        self.target_transform = target_transform
        self.img_dirs, self.label_dirs = self.get_img_dirs()

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the same image.
        """
        img = Image.open(self.img_dirs[idx]).convert("RGB")
        img = self.transform(img) if self.transform else img

        target = Image.open(self.label_dirs[idx]).convert("RGB")  # not sure mn el convert tbh
        target = self.target_transform(target) if self.target_transform else target
        return img, target

    def get_img_dirs(self):
        imgs = []
        labels = []
        for dir_path, sub_dirs, files in os.walk(self.dataset_path):
            for file in files:
                if (os.path.splitext(file)[1]).lower() in ['.png', '.jpg']:
                    if 'ask' in file:
                        labels.append(os.path.join(dir_path, file))
                    else:
                        imgs.append(os.path.join(dir_path, file))
        return imgs, labels


# Testing
DATASET_PATH = r"..\..\Dataset\In the Wild - RGB Segmentation"

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader

    wild_dataset = WildHandsDataset(DATASET_PATH, level=1)
    print(len(wild_dataset))
    data_loader = DataLoader(dataset=wild_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    data_iter = iter(data_loader)
    tensor_image, tensor_label = next(data_iter)
    print(tensor_image.shape)
    # todo imshow doesnt work
    # plt.imshow(tensor_image.permute(1, 2, 0))
    # plt.show()
