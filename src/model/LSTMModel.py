import torch
import torch.nn as nn
from model.LSTMLayer import LSTMLayer

class LSTMModel(nn.Module):
    """
    LSTM model for sequence processing.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    output_size : int
        Number of output features per time step.
    hidden_size : int
        Number of neurons in the hidden state.
    num_layers : int
        Number of recurrent layers (stacked GRUs).
    dropout : float
        Dropout probability between LSTM layers.

    Attributes
    ----------
    lstm : LSTMLayer
        The LSTM layer used for sequence modeling.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super(LSTMModel, self).__init__()
        self.lstm: LSTMLayer = LSTMLayer(input_size=input_size, 
                                      hidden_size=hidden_size, 
                                      num_layers=num_layers, 
                                      dropout=dropout)
        self.linear: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, output_size), representing
            the predicted values for each sequence in the batch.
        """
        out, h, c = self.lstm(x)

        out = self.linear(h[-1])

        return out