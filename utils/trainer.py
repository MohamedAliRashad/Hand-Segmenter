import torch


class Trainer:
    """
    Class to handle model Training.

    Args:

        model (callable): Neural Network for training.
        loss_fn (callable): A function to calculate the error. E.g, ``MSE, Cross Entropy``.
        optimizer (callable): Algorithim to backpropgate the loss. E.g, ``SGD, Adam``.
        device (object, optional): Choice between ``CPU`` & ``GPU`` (default: ``CPU``).
        save_dir (string, optional): folder to save training steps and final weights.
        is_progress_bar (bool, optional): enable/disable progress bar
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device=torch.device("cpu"),
        save_dir="result",
        is_progress_bar=True,
    ):
        pass

    def __call__(self, data_loader, epochs=10, checkpoint_every=10):
        pass
