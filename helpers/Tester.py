import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Tester:
    """
    Tester class for evaluating trained PyTorch models on unseen data.

    Parameters
    ----------
    model : nn.Module
        Trained neural network model to be evaluated.
    loss_fn : nn.Module
        Loss function used to assess prediction error.
    device : torch.device, optional
        Device on which to perform computations (CPU or GPU). 

    Attributes
    ----------
    model : nn.Module
        Model being evaluated.
    loss_fn : nn.Module
        Loss function used to compute test loss.
    device : torch.device
        Device used during evaluation.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> None:
        self.model: nn.Module = model 
        self.loss_fn: nn.Module = loss_fn
        self.device: torch.device = device
        
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Computes test loss and regression metrics.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the test dataset.

        Returns
        -------
        dict[str, float]
            Dictionary with loss, MSE, MAE, RMSE, and R2 scores.
        """

        preds: list[float] = []
        trues: list[float] = []
        total_loss: float = 0.0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch: torch.Tensor = X_batch.to(self.device)
                y_batch: torch.Tensor = y_batch.to(self.device)

                y_pred: torch.Tensor = self.model(X_batch)
                loss: torch.Tensor = self.loss_fn(y_pred.squeeze(), y_batch.float())

                total_loss += loss.item()
                preds.extend(y_pred.squeeze().cpu().numpy())
                trues.extend(y_batch.cpu().numpy())

        preds_array: np.ndarray = np.array(preds)
        trues_array: np.ndarray = np.array(trues)

        mse = mean_squared_error(trues_array, preds_array)
        mae = mean_absolute_error(trues_array, preds_array)
        rmse = np.sqrt(mse)
        r2 = r2_score(trues_array, preds_array)

        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Generates predictions for all samples in the given dataset.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the data to predict.

        Returns
        -------
        np.ndarray
            Array of model predictions.
        """

        predictions: list[int] = []

        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch: torch.Tensor = X_batch.to(self.device)
                y_pred: torch.Tensor = self.model(X_batch)

                predictions.extend(y_pred.squeeze().cpu().numpy())

        return np.array(predictions)