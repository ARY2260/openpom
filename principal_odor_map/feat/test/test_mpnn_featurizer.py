import numpy as np
from rdkit import Chem
import pytest
from principal_odor_map.feat.mpnn_featurizer \
    import atom_features, bond_features, GraphConvConstants


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
