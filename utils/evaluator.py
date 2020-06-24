import torch
from tqdm import tqdm
from texttable import Texttable


class Evaluator:
    """
    Class to handle model Evaluation.

    Args:
    ---
        __init__:
            model (nn.Module, callable): Neural Network for training.
            use_gpu (bool, optional): Choice between ``CPU`` & ``GPU`` (default: ``GPU`` if founded).
            is_progress_bar (bool, optional): enable/disable progress bar
        __call__:
            data_loaders (nn.DataLoader, list): either a single dataloader or a list of them to evaluate on
    """
    epsilon = 1e-9

    def __init__(self,
                 model,
                 use_gpu=True,
                 is_progress_bar=True):
        self.model = model
        self.is_progress_bar = is_progress_bar
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.mean_iou = 0
        self.mean_f_score = 0

    def __call__(self, data_loaders):
        table = Texttable()
        table.add_rows([["Dataset", "IoU (Jaccard)", "F1 score (Dice)"]])
        if type(data_loaders) is not list:
            data_loaders = [data_loaders]
        for loader in data_loaders:
            data_iter = iter(loader)
            total_iou = 0
            total_f_score = 0
            iterable = tqdm(data_iter) if self.is_progress_bar else data_iter
            for (img, ground_truth) in iterable:
                prediction = self.model(img)
                self.current_intersection = torch.sum(prediction * ground_truth)
                self.total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
                self.current_union = self.total_pixels - self.current_intersection
                total_iou += self.__iou()
                total_f_score += self.__f_score()
            mean_iou = total_iou / len(loader)
            mean_f_score = total_f_score / len(loader)
            dataset_name = type(loader.dataset).__name__.strip('Dataset')
            table.add_row([dataset_name, mean_iou, mean_f_score])
        print(table.draw())

    def __iou(self):
        return self.current_intersection / \
               (self.current_union + Evaluator.epsilon)

    def __f_score(self):
        return 2 * self.current_intersection / (self.total_pixels + Evaluator.epsilon)

    @staticmethod
    def iou(prediction, ground_truth):
        intersection = torch.sum(prediction * ground_truth)
        total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
        union = total_pixels - intersection
        return intersection / ((union - intersection) + Evaluator.epsilon)

    @staticmethod
    def f_score(prediction, ground_truth):
        intersection = torch.sum(prediction * ground_truth)
        total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
        return 2 * intersection / (total_pixels + Evaluator.epsilon)


if __name__ == "__main__":
    pass
