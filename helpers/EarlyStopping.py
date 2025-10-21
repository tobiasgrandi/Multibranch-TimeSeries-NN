from typing import Any
import torch
from torch import nn

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss 
    stops improving after a specified number of epochs (patience).

    Parameters
    ----------
    patience : int
        Number of consecutive epochs to wait for an improvement before stopping.
    min_delta : float, optional
        Minimum reduction in validation loss to qualify as an improvement.
    verbose : bool, optional
        If True, prints a message when early stopping is triggered. Default is False.
    path : str, optional
        Path to save the best model checkpoint. If None, the model is not saved.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum improvement threshold to reset counter.
    verbose : bool
        Whether to print messages when stopping early.
    path : str | None
        File path where the best model checkpoint is stored.
    best_loss : float
        Best validation loss recorded so far.
    counter : int
        Number of consecutive epochs without improvement.
    early_stop : bool
        Indicates whether early stopping should be triggered.
    """
    
    def __init__(self, patience: int, min_delta: float, verbose: bool = False, path: str | None = None) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.verbose: bool = verbose
        self.path: str | None = path

        self.best_loss: float = float('inf')
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Checks if validation loss has improved; otherwise increases the counter
        and sets early_stop to True if patience is exceeded.

        Parameters
        ----------
        val_loss : float
            Current validation loss from the latest epoch.
        model : nn.Module
            Model being trained, optionally saved if improvement occurs.

        Returns
        -------
        None
        """

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.path:
                torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f'Validation loss improved to {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose and (self.counter % 5 == 0 or self.counter == 1):
                print(f'No improvement in validation loss [{self.counter}/{self.patience}]')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")