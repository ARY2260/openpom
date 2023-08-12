from rdkit import Chem
from typing import List, Union, Dict, Sequence
from deepchem.utils.typing import RDKitAtom, RDKitBond
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_hybridization_one_hot
from principal_odor_map.utils.molecule_feature_utils \
    import get_atomic_num_one_hot, get_atom_total_valence_one_hot
import logging

logger = logging.getLogger(__name__)


class GraphConvConstants(object):
    """
    A class for holding featurization parameters.
    """

    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES: Dict[str, List[int]] = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-1, -2, 1, 2, 0],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
    }
    ATOM_FEATURES_HYBRIDIZATION: List[str] = [
        "SP", "SP2", "SP3", "SP3D", "SP3D2"
    ]
    # Dimension of atom feature vector
    ATOM_FDIM = sum(len(choices) + 1
                    for choices in ATOM_FEATURES.values()) + len(
                        ATOM_FEATURES_HYBRIDIZATION) + 1
    # len(choices) +1 and len(ATOM_FEATURES_HYBRIDIZATION)
    # + 1 to include room for unknown set
    BOND_FDIM = 6


def atom_features(atom: RDKitAtom) -> Sequence[Union[bool, int, float]]:
    """
    Helper method used to compute atom feature vector.

    Parameters
    ----------
    atom: RDKitAtom
        Atom to compute features on.

    Returns
    -------
    features: Sequence[Union[bool, int, float]]
        A list of atom features.
    """
    if atom is None:
        features: Sequence[Union[bool, int,
                                 float]] = [0] * GraphConvConstants.ATOM_FDIM

    else:
        features = []
        features += get_atom_total_valence_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['valence'])
        features += get_atom_total_degree_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['degree'])
        features += get_atom_total_num_Hs_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
        features += get_atom_formal_charge_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
        features += get_atomic_num_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
        features += get_atom_hybridization_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
        features = [int(feature) for feature in features]
    return features


def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:
    """
    Helper method used to compute bond feature vector.

    Parameters
    ----------
    bond: RDKitBond
        Bond to compute features on.

    Returns
    -------
    features: Sequence[Union[bool, int, float]]
        A list of bond features.
    """
    if bond is None:
        b_features: Sequence[Union[
            bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    else:
        bt = bond.GetBondType()
        b_features = [
            0, bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]

    return b_features
