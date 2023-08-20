from openpom.utils.molecule_feature_utils \
    import get_atomic_num_one_hot
from openpom.utils.molecule_feature_utils \
    import get_atom_total_valence_one_hot
from openpom.feat.graph_featurizer import GraphConvConstants
from rdkit import Chem
import numpy as np


def test_atomic_num_one_hot():
    """
    Test for get_atomic_num_one_hot feat util
    """
    smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    atom = m.GetAtoms()[0]
    f_atomic = get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    req_f = list(
        np.zeros((len(GraphConvConstants.ATOM_FEATURES['atomic_num']) + 1, ),
                 dtype=float))
    req_f[5] = 1.0
    assert len(f_atomic) == len(req_f)
    assert f_atomic == req_f


def test_total_valence_one_hot():
    """
    Test for get_atom_total_valence_one_hot feat util
    """
    smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    atom = m.GetAtoms()[0]
    f_valence = get_atom_total_valence_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['valence'])
    req_f = list(
        np.zeros((len(GraphConvConstants.ATOM_FEATURES['valence']) + 1, ),
                 dtype=float))
    req_f[4] = 1.0
    assert len(f_valence) == len(req_f)
    assert f_valence == req_f
