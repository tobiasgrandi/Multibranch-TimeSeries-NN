import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class YoutubeDataset(Dataset):
    """
    PyTorch Dataset for YouTube time series with sliding windows.

    This dataset creates fixed-length input sequences (windows) and their
    corresponding forecast targets from time series data of multiple videos.
    Each video is processed independently to preserve temporal consistency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data. Must include columns:
        ['ytvideoid', 'timestamp'] and the selected feature and target columns.
    features : list[str]
        List of feature column names used as input for each timestep.
    target : str
        Name of the column to be predicted.
    seq_len : int
        Number of timesteps in each input sequence.
    forecast_horizon : int
        Number of timesteps ahead to forecast.

    Attributes
    ----------
    sequences : torch.Tensor
        Tensor containing all input sequences with shape
        (num_sequences, seq_len, num_features).
    targets : torch.Tensor
        Tensor containing all target values with shape (num_sequences, 1).
    """

    def __init__(self, df: pd.DataFrame, features: list[str], target: str, seq_len: int, forecast_horizon: int) -> None:
        super().__init__()
        
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        grouped = df.groupby('ytvideoid')

        for video_id, group in grouped:
            group = group.sort_values('timestamp')
            X: np.ndarray = np.asarray(group[features].values)
            y: np.ndarray = np.asarray(group[target].values)

            for i in range(len(group) - seq_len - forecast_horizon + 1):
                x_seq: np.ndarray = X[i:i+seq_len]
                y_seq: np.ndarray = y[i+seq_len+forecast_horizon-1]
                sequences.append(x_seq)
                targets.append(y_seq) 

        self.sequences: torch.Tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.targets: torch.Tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.sequences)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single input-target pair.

        Parameters
        ----------
        index : int
            Index of the desired sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - Input tensor of shape (seq_len, num_features)
            - Target tensor of shape (1,)
        """
        return self.sequences[index], self.targets[index]