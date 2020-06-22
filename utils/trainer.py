import torch


class Trainer:
    """
    Class to handle model Training.

    Args:
    ---

        model (callable): Neural Network for training.
        loss_fn (callable): A function to calculate the error. E.g, ``MSE, Cross Entropy``.
        optimizer (callable): Algorithm to backpropgate the loss. E.g, ``SGD, Adam``.
        use_gpu (bool, optional): Choice between ``CPU`` & ``GPU`` (default: ``GPU`` if founded).
        save_dir (string, Path, optional): folder to save training steps and final weights.
        is_progress_bar (bool, optional): enable/disable progress bar
    """

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 use_gpu=True,
                 save_dir="result",
                 is_progress_bar=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    def __call__(self, data_loader, epochs=10, checkpoint_every=10):
        # checkpoint ??
        for epoch in range(epochs):
            running_loss = 0.0
            for index, (img, ground_truth) in enumerate(data_loader):
                self.optimizer.zero_grad()
                outputs = self.model(img)
                loss = self.loss_fn(outputs, ground_truth)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if index % 100 == 0:  # print every 100 mini-batches
                    print(f'[{epoch + 1}, {index + 1}] loss: {running_loss / 100:.3}')
                    running_loss = 0
