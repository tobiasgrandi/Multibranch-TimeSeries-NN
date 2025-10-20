import torch
import torch.nn as nn
import numpy as np
from LSTM import LSTMModel
from GRU import GRUModel

np.random.seed(0)
torch.manual_seed(0)

class MultiBranchModel(nn.Module):
    """
    Multi-branch model combining LSTM and GRU for sequence forecasting.

    This model processes input sequences through two parallel branches:
    one LSTM branch and one GRU branch. The final hidden states of both
    branches are concatenated and passed through a linear layer to
    produce the output prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : dict[str, int]
        Dictionary specifying hidden state sizes for each branch.
        Example: {'lstm': 128, 'gru': 64}.
    num_layers : dict[str, int]
        Dictionary specifying number of layers for each branch.
        Example: {'lstm': 2, 'gru': 1}.
    dropout : dict[str, float]
        Dictionary specifying dropout probability for each branch.
        Example: {'lstm': 0.2, 'gru': 0.1}.
    output_size : int
        Dimensionality of the model output (e.g., number of predicted values).

    Attributes
    ----------
    lstm : LSTMModel
        The LSTM branch used for sequence modeling.
    gru : GRUModel
        The GRU branch used for sequence modeling.
    linear : nn.Linear
        Linear layer that combines LSTM and GRU outputs into final predictions.
    """
        
    def __init__(self, input_size: int, 
                 hidden_size: dict[str, int], 
                 num_layers: dict[str, int], 
                 dropout: dict[str, int], 
                 output_size: int) -> None:
        super(MultiBranchModel).__init__()
        self.lstm: LSTMModel = LSTMModel(input_size, hidden_size['lstm'], num_layers['lstm'], dropout['lstm'])
        self.gru: GRUModel = GRUModel(input_size, hidden_size['gru'], num_layers['gru'], dropout['gru'])
        self.linear = nn.Linear(hidden_size['lstm'] + hidden_size['gru'], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-branch model.

        Processes the input sequence through the LSTM and GRU branches,
        concatenates their final hidden states, and passes the result
        through the linear layer to produce the output.

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

        _, (hn_lstm, _) = self.lstm(x)

        lstm_out = hn_lstm[-1]

        _, hn_gru = self.gru(x)

        gru_out = hn_gru[-1]

        combined = torch.cat((lstm_out, gru_out), dim=1)

        out = self.linear(combined)
        return out