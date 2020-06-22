import torch


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

    def __init__(
        self,
        model,
        use_gpu=True,
        is_progress_bar=True,
    ):
        pass

    def __call__(self, data_loaders):
        pass
