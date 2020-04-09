import torch


class Evaluator:
    """
    Class to handle model Evaluation.

    Args:

        model (callable): Neural Network for training.
        device (object, optional): Choice between ``CPU`` & ``GPU`` (default: ``CPU``).
        is_progress_bar (bool, optional): enable/disable progress bar
    """

    def __init__(
        self,
        model,
        device=torch.device("cpu"),
        is_progress_bar=True,
    ):
        pass

    def __call__(self, data_loader):
        pass
