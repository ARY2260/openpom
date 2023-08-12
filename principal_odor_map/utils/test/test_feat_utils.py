from principal_odor_map.utils.molecule_feature_utils \
    import get_atomic_num_one_hot
from principal_odor_map.feat.mpnn_featurizer import GraphConvConstants
from rdkit import Chem
import numpy as np


def test_atomic_num_one_hot():
    """
    Test for _get_atomic_num_one_hot feat util
    """
    smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    atom = m.GetAtoms()[0]
    f_atomic = get_atomic_num_one_hot(
        atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
    req_f = list(np.zeros((101, ), dtype=float))
    req_f[5] = 1.0
    assert len(f_atomic) == len(req_f)
    assert f_atomic == req_f
