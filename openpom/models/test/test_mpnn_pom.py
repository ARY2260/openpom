import torch
import dgl
import pytest
import numpy as np
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOM


def test_mpnnpom_regression():
    """
    Test MPNNPOM class for regression
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    input_smile = ["CC", "C"]
    feat = GraphFeaturizer()
    graphs = feat.featurize(input_smile)
    dgl_graphs = [graph.to_dgl_graph() for graph in graphs]
    g = dgl.batch(dgl_graphs).to(device)

    number_of_molecules = 2
    n_tasks = 3
    model = MPNNPOM(n_tasks=n_tasks,
                    mode='regression',
                    number_atom_features=134,
                    number_bond_features=6)

    # assert layers
    model.to(device)
    output = model(g)
    assert output.shape == torch.Size([number_of_molecules, n_tasks])

    required_output = np.asarray([[0.0687, 0.7174, 0.0861],
                                  [-0.1467, 0.0957, 0.1467]])
    assert np.allclose(output.detach().cpu().numpy(),
                       required_output,
                       atol=0.001)


# Set up testing parameters.
Test1_params = {
    'mpnn_residual': True,
    'message_aggregator_type': 'sum',
    'readout_type': 'set2set'
}

Test2_params = {
    'mpnn_residual': False,
    'message_aggregator_type': 'mean',
    'readout_type': 'global_sum_pooling'
}

Test3_params = {
    'mpnn_residual': True,
    'message_aggregator_type': 'max',
    'readout_type': 'set2set'
}


@pytest.mark.parametrize('test_parameters',
                         [Test1_params, Test2_params, Test3_params])
def test_mpnnpom_multiple_configs(test_parameters):
    """
    Test MPNNPOM class for multiple configs
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    input_smile = ["CC", "C"]
    feat = GraphFeaturizer()
    graphs = feat.featurize(input_smile)
    dgl_graphs = [graph.to_dgl_graph() for graph in graphs]
    g = dgl.batch(dgl_graphs).to(device)

    number_of_molecules = 2
    n_tasks = 3

    mpnn_residual, message_aggregator_type, readout_type = \
        test_parameters.values()
    model = MPNNPOM(n_tasks=n_tasks,
                    mode='regression',
                    number_atom_features=134,
                    number_bond_features=6,
                    mpnn_residual=mpnn_residual,
                    message_aggregator_type=message_aggregator_type,
                    readout_type=readout_type)

    # assert layers
    model.to(device)
    output = model(g)
    assert output.shape == torch.Size([number_of_molecules, n_tasks])


def test_mpnnpom_classification_single_task():
    """
    Test MPNNPOM class for single task classification
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    input_smile = ["CC", "C"]
    feat = GraphFeaturizer()
    graphs = feat.featurize(input_smile)
    dgl_graphs = [graph.to_dgl_graph() for graph in graphs]
    g = dgl.batch(dgl_graphs).to(device)

    number_of_molecules = 2
    n_tasks = 1
    n_classes = 1
    embedding_size = 2
    model = MPNNPOM(n_tasks=n_tasks,
                    mode='classification',
                    number_atom_features=134,
                    number_bond_features=6,
                    n_classes=n_classes,
                    ffn_embeddings=embedding_size)

    # assert layers
    model.to(device)
    output = model(g)
    assert len(output) == 3
    assert output[0].shape == torch.Size([number_of_molecules])
    assert output[1].shape == torch.Size([number_of_molecules, n_classes])
    assert output[2].shape == torch.Size([number_of_molecules, embedding_size])

    required_output0 = np.asarray([0.2934, 0.4143])
    required_output1 = np.asarray([[-0.8789], [-0.3463]])
    required_output2 = np.asarray([[0.3893, 0.1915], [0.0096, -0.0100]])
    assert np.allclose(output[0].detach().cpu().numpy(),
                       required_output0,
                       atol=0.001)
    assert np.allclose(output[1].detach().cpu().numpy(),
                       required_output1,
                       atol=0.001)
    assert np.allclose(output[2].detach().cpu().numpy(),
                       required_output2,
                       atol=0.001)


def test_mpnnpom_classification_multi_task():
    """
    Test MPNNPOM class for multi task classification
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    input_smile = ["CC", "C"]
    feat = GraphFeaturizer()
    graphs = feat.featurize(input_smile)
    dgl_graphs = [graph.to_dgl_graph() for graph in graphs]
    g = dgl.batch(dgl_graphs).to(device)

    number_of_molecules = 2
    n_tasks = 3
    n_classes = 1
    embedding_size = 2
    model = MPNNPOM(n_tasks=n_tasks,
                    mode='classification',
                    number_atom_features=134,
                    number_bond_features=6,
                    n_classes=n_classes,
                    ffn_embeddings=embedding_size)

    # assert layers
    model.to(device)
    output = model(g)
    assert len(output) == 3
    assert output[0].shape == torch.Size([number_of_molecules, n_tasks])
    assert output[1].shape == torch.Size(
        [number_of_molecules, n_tasks, n_classes])
    assert output[2].shape == torch.Size([number_of_molecules, embedding_size])

    required_output0 = np.asarray([[0.2934, 0.5467, 0.3940],
                                   [0.4143, 0.6249, 0.3807]])
    required_output1 = np.asarray([[[-0.8789], [0.1873], [-0.4306]],
                                   [[-0.3463], [0.5105], [-0.4868]]])
    required_output2 = np.asarray([[0.3893, 0.1915], [0.0096, -0.0100]])
    assert np.allclose(output[0].detach().cpu().numpy(),
                       required_output0,
                       atol=0.001)
    assert np.allclose(output[1].detach().cpu().numpy(),
                       required_output1,
                       atol=0.001)
    assert np.allclose(output[2].detach().cpu().numpy(),
                       required_output2,
                       atol=0.001)
