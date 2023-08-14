import torch.nn as nn
from dgl.nn.pytorch import NNConv
from dgllife.model.gnn import MPNNGNN


class CustomMPNNGNN(MPNNGNN):
    """
    Customized MPNNGNN layer based MPNNGNN layer in dgllife library.

    Additional options:
    -> toggle for residual in gnn layer
    -> choice for message aggregator type

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN
    and returns the updated node representations.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        """
        Parameters
        ----------
        node_in_feats: int
            Size for the input node features.
        node_out_feats: int
            Size for the output node representations. Default to 64.
        edge_in_feats: int
            Size for the input edge features. Default to 128.
        edge_hidden_feats: int
            Size for the hidden edge representations.
        num_step_message_passing: int
            Number of message passing steps. Default to 6.
        residual: bool
            If true, adds residual layer to gnn layer
        message_aggregator_type: str
            message aggregator type, 'sum', 'mean' or 'max'
        """
        super(CustomMPNNGNN,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        self.gnn_layer = NNConv(in_feats=node_out_feats,
                                out_feats=node_out_feats,
                                edge_func=edge_network,
                                aggregator_type=message_aggregator_type,
                                residual=residual)
