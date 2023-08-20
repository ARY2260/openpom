from typing import List
from deepchem.utils.typing import RDKitAtom
from deepchem.utils.molecule_feature_utils import one_hot_encode


def get_atomic_num_one_hot(atom: RDKitAtom,
                           allowable_set: List[int],
                           include_unknown_set: bool = True) -> List[float]:
    """
    Get a one-hot feature about atomic number of the given atom.

    Parameters
    ---------
    atom: RDKitAtom
        RDKit atom object
    allowable_set: List[int]
        The range of atomic numbers to consider.
    include_unknown_set: bool, default False
        If true, the index of all types not in
        `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of atomic number of the given atom.
        If `include_unknown_set` is False, the length is
        `len(allowable_set)`.
        If `include_unknown_set` is True, the length is
        `len(allowable_set) + 1`.

    """
    return one_hot_encode(atom.GetAtomicNum() - 1, allowable_set,
                          include_unknown_set)


def get_atom_total_valence_one_hot(
        atom: RDKitAtom,
        allowable_set: List[int],
        include_unknown_set: bool = True) -> List[float]:
    """Get a one-hot feature for total valence of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        Atom total valence to consider.
    include_unknown_set: bool, default True
        If true, the index of all types not in
        `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector for total valence an atom has.
        If `include_unknown_set` is False, the length is
        `len(allowable_set)`.
        If `include_unknown_set` is True, the length is
        `len(allowable_set) + 1`.

    """
    return one_hot_encode(atom.GetTotalValence(), allowable_set,
                          include_unknown_set)
