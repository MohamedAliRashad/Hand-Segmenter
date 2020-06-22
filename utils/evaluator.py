import torch
from tqdm import tqdm


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
            total_iou += self.iou()
            total_f_score += self.f_score()
        return total_iou / len(data_loader), total_f_score / len(data_loader)

    def iou(self):
        return self.current_intersection / \
               ((self.current_union - self.current_intersection) + self.epsilon)

    def f_score(self):
        return 2 * self.current_intersection / (self.total_pixels + self.epsilon)


if __name__ == "__main__":
    pass
