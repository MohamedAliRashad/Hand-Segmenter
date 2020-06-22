import torch
import sys
import numpy as np


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
        self.use_gpu = use_gpu
        self.is_progress_bar = is_progress_bar
        self.epsilon = epsilon
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    def __call__(self, data_loader):
        data_iter = iter(data_loader)
        total_iou = 0
        total_f_score = 0
        self.init_progress_bar()
        for index, (img, ground_truth) in enumerate(data_iter):
            img_tensor = self.img_to_tensor(img)
            prediction = self.model.forward(img_tensor)
            self.current_intersection = torch.sum(prediction * ground_truth)
            self.total_pixels = torch.sum(ground_truth) + torch.sum(prediction)
            self.current_union = self.total_pixels - self.current_intersection
            total_iou += self.iou()
            total_f_score += self.f_score()
            self.print_progress(index, len(data_loader))
        return total_iou / len(data_loader), total_f_score / len(data_loader)

    def iou(self):
        return self.current_intersection / \
               ((self.current_union - self.current_intersection) + self.epsilon)

    def f_score(self):
        return 2 * self.current_intersection / (self.total_pixels + self.epsilon)

    def print_progress(self, index, length):
        if self.is_progress_bar:
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("[")
            sys.stdout.write("#" * (index + 1))
            sys.stdout.write("_" * (length - index - 1))
            sys.stdout.write(f"]")
            sys.stdout.write(f" {int((index + 1) / length * 100)}%")
            sys.stdout.write(f" {(index + 1)}/{length} items")
            sys.stdout.flush()

    def init_progress_bar(self):
        # each hash represents 2 % of the progress
        if self.is_progress_bar:
            toolbar_width = 50
            sys.stdout.write(f"[{'_' * toolbar_width}]")
            sys.stdout.flush()

    def img_to_tensor(self, img):
        img_tensor = torch.from_numpy(np.array(img))  # not sure if this is correct/optimal
        img_tensor.to(self.device)
        return img_tensor


if __name__ == "__main__":
    pass
