import numpy as np
from rdkit import Chem
import pytest
from openpom.feat.graph_featurizer \
    import atom_features, bond_features, GraphConvConstants, GraphFeaturizer


# Tests for helper functions
@pytest.fixture
def example_smiles_n_features():
    """
    Sample data for testing

    Returns
    -------
    dictionary
    format {'smiles':required feature vector : List}
    """
    feature_vector_C = [[
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    ]]
    feature_vector_NN = [[
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    ],
                         [
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
                         ]]
    return {'C': feature_vector_C, 'N#N': feature_vector_NN}


@pytest.fixture
def example_smiles_n_b_features():
    """
    Sample data for testing

    Returns
    -------
    dictionary
    format {'smiles':required feature vector}
    """
    feature_vector_C1OC1 = [[0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 1]]
    feature_vector_NN = [[0, 0, 0, 1, 0, 0]]
    return {'C1OC1': feature_vector_C1OC1, 'N#N': feature_vector_NN}


def test_atom_features_none():
    """
    Test for atom_features() with 'None' input for Atom value
    """
    f_atom = atom_features(None)
    req_f = list(np.zeros((GraphConvConstants.ATOM_FDIM, ), dtype=int))
    assert len(f_atom) == len(req_f)
    assert f_atom == req_f


def test_atom_features(example_smiles_n_features):
    """
    Test for atom_features() function
    """
    for smiles in example_smiles_n_features.keys():
        m = Chem.MolFromSmiles(smiles)
        f = []
        for atom in m.GetAtoms():
            features = atom_features(atom)
            f.append(features)
        k = np.array(f)
        req_f = np.array(example_smiles_n_features[smiles])
        assert k.shape == req_f.shape
        assert f == example_smiles_n_features[smiles]


def test_bond_features_none():
    """
    Test for bond_features() with 'None' input for bond
    """
    f_bond = bond_features(None)
    req_f = list(np.zeros((GraphConvConstants.BOND_FDIM, ), dtype=int))
    req_f[0] = 1
    assert len(f_bond) == len(req_f)
    assert f_bond == req_f


def test_bond_features(example_smiles_n_b_features):
    """
    Test for bond_features() function
    """
    for smiles in example_smiles_n_b_features.keys():
        b_f = []
        m = Chem.MolFromSmiles(smiles)
        for b in m.GetBonds():
            b_f.append(bond_features(b))
        print(b_f)
        k = np.array(b_f)
        req_f = np.array(example_smiles_n_b_features[smiles])
        assert k.shape == req_f.shape
        assert b_f == example_smiles_n_b_features[smiles]


# Tests for graph featurizer

required_edge_index = {
    "C1=CC=NC=C1":
    np.asarray([[0, 1, 1, 5, 5, 2, 2, 4, 4, 3, 3, 0],
                [1, 0, 5, 1, 2, 5, 4, 2, 3, 4, 0, 3]]),
    "CC(=O)C":
    np.asarray([[0, 3, 3, 2, 3, 1], [3, 0, 2, 3, 1, 3]]),
    "C":
    np.empty((2, 0), dtype=int)
}


def test_graph_featurizer_single_atom():
    """
    Test for featurization of "C" using `GraphFeaturizer` class.
    """
    smiles = "C"
    featurizer = GraphFeaturizer()  # is_adding_hs=False
    graph_feat = featurizer.featurize(smiles)
    assert graph_feat[0].num_nodes == 1
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 0
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (0,
                                                 GraphConvConstants.BOND_FDIM)
    assert graph_feat[0].edge_index.shape == required_edge_index['C'].shape
    assert (graph_feat[0].edge_index == required_edge_index['C']).all()


def test_graph_featurizer_general_case():
    """
    Test for featurization of "CC(=O)C" using `GraphFeaturizer` class.
    """
    smiles = "CC(=O)C"
    featurizer = GraphFeaturizer()  # is_adding_hs=False
    graph_feat = featurizer.featurize(smiles)
    assert graph_feat[0].num_nodes == 4
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (4,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 6
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (6,
                                                 GraphConvConstants.BOND_FDIM)
    assert graph_feat[0].edge_index.shape == required_edge_index[
        'CC(=O)C'].shape
    assert (graph_feat[0].edge_index == required_edge_index['CC(=O)C']).all()


def test_graph_featurizer_ring():
    """
    Test for featurization of "C1=CC=NC=C1" using `GraphFeaturizer` class.
    """
    smiles = "C1=CC=NC=C1"
    featurizer = GraphFeaturizer()  # is_adding_hs=False
    graph_feat = featurizer.featurize(smiles)
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (12,
                                                 GraphConvConstants.BOND_FDIM)
    assert graph_feat[0].edge_index.shape == required_edge_index[
        'C1=CC=NC=C1'].shape
    assert (
        graph_feat[0].edge_index == required_edge_index['C1=CC=NC=C1']).all()
