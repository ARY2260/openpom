import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any


class CustomPositionwiseFeedForward(nn.Module):
    """
    Customised PositionwiseFeedForward layer from deepchem
    for:
        - hidden layers of variable sizes
        - batch normalization before every activation function
        - additional output of embedding layer (penultimate layer)
          for POM embeddings.
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_hidden_list: List = [1024],
        d_output: int = 1024,
        activation: str = 'leakyrelu',
        dropout_p: float = 0.0,
        dropout_at_input_no_act: bool = False,
        batch_norm: bool = True,
    ):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden_list: List
            List of hidden sizes.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU,
            'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU
            and 'linear' for linear activation.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor.
            For single layer, it is not passed to an activation function.
        batch_norm: bool
            If true, applies batch normalization
            'before' every activation function
        """
        super(CustomPositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act
        self.batch_norm: bool = batch_norm

        self.activation: Callable[[Any], Any]
        if activation == 'relu':
            self.activation = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == 'linear':
            self.activation = lambda x: x

        d_output = d_output if d_output != 0 else d_input

        # Set n_layers
        self.n_layers: int = len(d_hidden_list) + 1

        # Set linear layers
        if self.n_layers == 1:
            linears: List = [nn.Linear(d_input, d_output)]

        else:
            linears = [nn.Linear(d_input, d_hidden_list[0])]
            for idx in range(1, len(d_hidden_list)):
                linears.append(
                    nn.Linear(d_hidden_list[idx - 1], d_hidden_list[idx]))
            linears.append(nn.Linear(d_hidden_list[-1], d_output))

        self.linears: nn.ModuleList = nn.ModuleList(linears)
        dropout_layer: nn.Dropout = nn.Dropout(dropout_p)
        self.dropout_p: nn.ModuleList = nn.ModuleList(
            [dropout_layer for _ in range(self.n_layers)])

        if batch_norm:
            batchnorms: List = [
                nn.BatchNorm1d(d_hidden_list[idx])
                for idx in range(len(d_hidden_list))
            ]
            self.batchnorms: nn.ModuleList = nn.ModuleList(batchnorms)

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """
        Output Computation for the Customised
        PositionwiseFeedForward layer

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        List[Optional[torch.Tensor]]
            List containing embeddings and output
        """

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return [None, self.linears[0](self.dropout_p[0](x))]
            else:
                return [
                    None,
                    self.dropout_p[0](self.activation(self.linears[0](x)))
                ]

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[-1](x)

            if self.batch_norm:
                for i in range(self.n_layers - 2):
                    x = self.dropout_p[i](self.activation(self.batchnorms[i](
                        self.linears[i](x))))

                embeddings: torch.Tensor = self.linears[self.n_layers - 2](x)
                x = self.dropout_p[self.n_layers - 2](self.activation(
                    self.batchnorms[self.n_layers - 2](embeddings)))
            else:
                for i in range(self.n_layers - 2):
                    x = self.dropout_p[i](self.activation(self.linears[i](x)))

                embeddings = self.linears[self.n_layers - 2](x)
                x = self.dropout_p[self.n_layers - 2](
                    self.activation(embeddings))

            output: torch.Tensor = self.linears[-1](x)
            return [embeddings, output]
