import torch
import torch.nn as nn
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

class LSTMModel(nn.Module):
    """
    LSTM model for sequence processing.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of neurons in the hidden state.
    num_layers : int
        Number of recurrent layers (stacked LSTMs).
    output_size : int
        Number of output features.
    dropout : float
        Dropout probability between LSTM layers.

    Attributes
    ----------
    hidden_size : int
        Hidden state dimensionality.
    num_layers : int
        Number of LSTM layers.
    lstm : nn.LSTM
        The LSTM module used for sequence modeling.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.lstm: nn.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor, 
                h0: torch.Tensor | None = None, 
                c0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).
        h0 : torch.Tensor | None, optional
            Initial hidden state of shape (num_layers, batch_size, hidden_size).
            If None, a zero tensor is used.
        c0 : torch.Tensor | None, optional
            Initial cell state of shape (num_layers, batch_size, hidden_size).
            If None, a zero tensor is used.

        Returns
        -------
        out : torch.Tensor
            Output features from the LSTM for each time step, 
            of shape (batch_size, seq_len, hidden_size).
        hn : torch.Tensor
            Hidden state for the last time step,
            of shape (num_layers, batch_size, hidden_size).
        cn : torch.Tensor
            Cell state for the last time step,
            of shape (num_layers, batch_size, hidden_size).
        """
                
        batch_size = x.size(0)
        device = x.device
        hidden_state: torch.Tensor = h0 if h0 is not None else torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell_state: torch.Tensor = c0 if c0 is not None else torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (hidden_state, cell_state))
        return out, hn, cn