import torch
from tqdm import tqdm
from texttable import Texttable


class Evaluator:
    """
    Class to handle model Evaluation.

    Args:
    ---
        model (callable): Neural Network for training.
        use_gpu (bool, optional): Choice between ``CPU`` & ``GPU`` (default: ``GPU`` if founded).
        is_progress_bar (bool, optional): enable/disable progress bar
        epsilon  (float, optional): to avoid division by zero errors
    """

    def __init__(self,
                 model,
                 use_gpu=True,
                 is_progress_bar=True,
                 epsilon=1e-9):
        self.model = model
        self.is_progress_bar = is_progress_bar
        self.epsilon = epsilon
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.mean_iou = 0
        self.mean_f_score = 0
        self.table = Texttable()
        self.table.add_rows([["Dataset", "IoU (Jaccard)", "F1 score (Dice)"]])

    def __call__(self, data_loader):
        data_iter = iter(data_loader)
        total_iou = 0
        total_f_score = 0
        iterable = tqdm(enumerate(data_iter)) if self.is_progress_bar else enumerate(data_iter)
        for index, (img, ground_truth) in iterable:
            prediction = self.model(img)
            self.current_intersection = torch.sum(prediction * ground_truth)
            self.total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
            self.current_union = self.total_pixels - self.current_intersection
            total_iou += self.__iou()
            total_f_score += self.__f_score()
        self.mean_iou = total_iou / len(data_loader)
        self.mean_f_score = total_f_score / len(data_loader)
        self.dataset_name = type(data_loader.dataset).__name__
        self.add_row_to_table()

    def __iou(self):
        return self.current_intersection / \
               ((self.current_union - self.current_intersection) + self.epsilon)

    def __f_score(self):
        return 2 * self.current_intersection / (self.total_pixels + self.epsilon)

    def add_row_to_table(self):
        self.table.add_rows([["", self.mean_iou, self.mean_f_score]])
        print(self.table.draw())

    def print_metrics(self):
        print(self.table.draw())

    @staticmethod
    def iou(prediction, ground_truth, epsilon=1e-9):
        intersection = torch.sum(prediction * ground_truth)
        total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
        union = total_pixels - intersection
        return intersection / ((union - intersection) + epsilon)

    @staticmethod
    def f_score(prediction, ground_truth, epsilon=1e-9):
        intersection = torch.sum(prediction * ground_truth)
        total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
        return 2 * intersection / (total_pixels + epsilon)


if __name__ == "__main__":
    pass
