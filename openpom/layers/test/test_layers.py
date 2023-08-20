import torch
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.layers.pom_ffn import CustomPositionwiseFeedForward
from openpom.layers.pom_mpnn_gnn import CustomMPNNGNN


def test_custom_position_wise_feed_forward():
    """Test invoking CustomPositionwiseFeedForward."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    ffn = CustomPositionwiseFeedForward(d_input=2,
                                        d_hidden_list=[16, 3],
                                        d_output=2,
                                        activation='leakyrelu',
                                        dropout_p=0.1,
                                        dropout_at_input_no_act=True,
                                        batch_norm=True)
    assert len(ffn.batchnorms) == 2
    embbedding_result, output_result = ffn(input_ar)
    assert embbedding_result.shape == (2, 3)
    assert output_result.shape == (2, 2)


def test_custom_position_wise_feed_forward_no_batchnorm():
    """Test invoking CustomPositionwiseFeedForward without batch norm"""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    ffn = CustomPositionwiseFeedForward(d_input=2,
                                        d_hidden_list=[16, 3],
                                        d_output=2,
                                        activation='leakyrelu',
                                        dropout_p=0.1,
                                        dropout_at_input_no_act=True,
                                        batch_norm=False)
    assert not hasattr(ffn, 'batchnorms')
    embbedding_result, output_result = ffn(input_ar)
    assert embbedding_result.shape == (2, 3)
    assert output_result.shape == (2, 2)


def test_custom_mpnn_gnn_residual_sum():
    """
    Test invoking CustomMPNNGNN with residual
    and message summation
    """
    torch.manual_seed(0)
    mpnngnn = CustomMPNNGNN(node_in_feats=134,
                            edge_in_feats=6,
                            node_out_feats=4,
                            edge_hidden_feats=10,
                            num_step_message_passing=3,
                            residual=True,
                            message_aggregator_type='sum')
    assert mpnngnn.gnn_layer.res_fc is not None
    assert mpnngnn.gnn_layer.reducer.__name__ == 'sum'

    featurizer = GraphFeaturizer()
    graph = featurizer.featurize('O=C=O')[0]
    g = graph.to_dgl_graph(self_loop=False)

    node_feats = g.ndata['x']
    edge_feats = g.edata['edge_attr']

    node_encodings = mpnngnn(g, node_feats, edge_feats)
    assert node_encodings.shape == (3, 4)


def test_custom_mpnn_gnn_no_residual_mean():
    """
    Test invoking CustomMPNNGNN with no residual
    and message mean aggregation
    """
    torch.manual_seed(0)
    mpnngnn1 = CustomMPNNGNN(node_in_feats=134,
                             edge_in_feats=6,
                             node_out_feats=10,
                             edge_hidden_feats=10,
                             num_step_message_passing=3,
                             residual=False,
                             message_aggregator_type='mean')
    assert mpnngnn1.gnn_layer.res_fc is None
    assert mpnngnn1.gnn_layer.reducer.__name__ == 'mean'

    featurizer = GraphFeaturizer()
    graph = featurizer.featurize('O=C=O')[0]
    g = graph.to_dgl_graph(self_loop=False)

    node_feats = g.ndata['x']
    edge_feats = g.edata['edge_attr']

    node_encodings1 = mpnngnn1(g, node_feats, edge_feats)
    assert node_encodings1.shape == (3, 10)
