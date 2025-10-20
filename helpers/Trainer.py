import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    """
    Trainer class for managing the training and validation process of PyTorch models.

    Parameters
    ----------
    model : nn.Module
        Neural network model to be trained.
    optimizer : torch.optim.Optimizer
        Optimization algorithm used.
    loss_fn : nn.Module
        Loss function to minimize during training.
    device : torch.device
        Device on which to perform computations (CPU or GPU). 

    Attributes
    ----------
    model : nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for gradient descent.
    loss_fn : nn.Module
        The loss function guiding the optimization process.
    device : torch.device
        The computation device used during training and validation.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> None:
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: nn.Module = loss_fn
        self.device: torch.device = device

        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Executes one full training epoch over the provided dataset.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of training data (inputs and targets).

        Returns
        -------
        float
            Average training loss computed over all batches in the epoch.
        """

        self.model.train()
        total_loss: float = 0.0

        for X_batch, y_batch in tqdm(dataloader, desc='Training', leave=False):

            X_batch: torch.Tensor = X_batch.to(self.device)
            y_batch: torch.Tensor = y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred.squeeze(), y_batch.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Evaluates the model performance on a validation dataset.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of validation data (inputs and targets).

        Returns
        -------
        float
            Average validation loss across all batches.
        """
        
        self.model.eval()
        total_loss: float  = 0.0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch: torch.Tensor = X_batch.to(self.device)
                y_batch: torch.Tensor = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred.squeeze(), y_batch.float())
                total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """
        Trains the model for a specified number of epochs, optionally evaluating on a validation set.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing batches of training data.
        val_loader : DataLoader
            DataLoader for validation data.
        epochs : int
            Number of epochs to train the model.

        Returns
        -------
        None
            This method does not return a value; it trains the model in place.
        """
                
        for epoch in range(epochs):
            train_loss: float = self.train_epoch(train_loader)
            val_loss: float = self.validate_epoch(val_loader)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')