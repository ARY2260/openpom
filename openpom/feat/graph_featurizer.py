import numpy as np
from rdkit import Chem
from typing import List, Union, Dict, Sequence
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils \
    import get_atom_hybridization_one_hot
from openpom.utils.molecule_feature_utils \
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


class GraphFeaturizer(MolecularFeaturizer):
    """
    This class is a featurizer for GNN (MESSAGE PASSING) implementation for
    Principal Odor Map.

    The default node(atom) and edge(bond) representations are based on
    `A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception
    preprint <https://www.biorxiv.org/content/10.1101/2022.09.01.504602v4>`_.

    The default node representation are constructed by concatenating
    the following values, and the feature length is 134.

    - Valence: A one-hot vector for total valence (0-6) of an atom.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens
      (0-4) that this atom connected.
    - Formal charge: Integer electronic charge, -1, -2, 1, 2, 0.
    - Atomic num: A one-hot vector of this atom, in a range of first 100 atoms.
    - Hybridization: A one-hot vector of "SP", "SP2", "SP3", "SP3D", "SP3D2".

    The default edge representation are constructed by concatenating
    the following values, and the feature length is 6.

    - Bond type: A one-hot vector of the bond type,
      "single", "double", "triple", or "aromatic".
    - Is in ring: Boolean value to specify whether
      the bond is in a ring or not.

    If you want to know more details about features,
    please check the paper [1]_ and utilities in
    deepchem.utils.molecule_feature_utils.py.

    References
    ----------
    .. [1] Kearnes, Steven, et al.
       "Molecular graph convolutions: moving beyond fingerprints."
        Journal of computer-aided molecular design 30.8 (2016):595-608.

    Note
    ----
    This class requires RDKit to be installed.

    """

    def __init__(self, is_adding_hs=False):
        """
        Parameters
        ----------
        is_adding_hs: bool, default False
            Whether to add Hs or not.
        """
        self.is_adding_hs = is_adding_hs
        super(GraphFeaturizer).__init__()

    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Construct edge (bond) index

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        edge_index: np.ndarray
            Edge (Bond) index

        """
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph object with features:
            - node_features: Node feature matrix with shape
              [num_nodes, num_node_features]
            - edge_index: Graph connectivity in COO format with shape
              [2, num_edges]
            - edge_features: Edge feature matrix with shape
              [num_edges, num_edge_features]
        """
        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError(
                "Feature field should contain smiles for featurizer!")

        # get atom features
        f_atoms: np.ndarray = np.asarray(
            [atom_features(atom) for atom in datapoint.GetAtoms()],
            dtype=float)

        # get edge(bond) features
        if len(datapoint.GetBonds()) == 0:
            f_bonds: np.ndarray = np.empty((0, GraphConvConstants.BOND_FDIM))
        else:
            f_bonds_list = []
            for bond in datapoint.GetBonds():
                b_feat = 2 * [bond_features(bond)]
                f_bonds_list.extend(b_feat)
            f_bonds = np.asarray(f_bonds_list, dtype=float)

        # get edge index
        edge_index: np.ndarray = self._construct_bond_index(datapoint)

        return GraphData(node_features=f_atoms,
                         edge_index=edge_index,
                         edge_features=f_bonds)
