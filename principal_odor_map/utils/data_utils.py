import pandas as pd
import numpy as np
from typing import List
from deepchem.data.datasets import DiskDataset


def get_class_imbalance_ratio(dataset: DiskDataset) -> List:
    """
    Get imbalance ratio per task from DiskDataset

    Imbalance ratio per label (IRLbl): Let M be an MLD with a set of
    labels L and Yi be the label-set of the ith instance. IRLbl is calcu-
    lated for the label λ as the ratio between the majority label and
    the label λ, where IRLbl is 1 for the most frequent label and a
    greater value for the rest. The larger the value of IRLbl, the higher
    the imbalance level for the concerned label.

    Parameters
    ---------
    dataset: DiskDataset
        Deepchem diskdataset object to get class imbalance ratio

    Returns
    -------
    class_imbalance_ratio: List
        List of imbalance ratios per task

    References
    ----------
    .. TarekegnA.N. et al.
       "A review of methods for imbalanced multi-label classification"
       Pattern Recognit. (2021)
    """
    if not isinstance(dataset, DiskDataset):
        raise Exception("The dataset should be a deepchem DiskDataset")
    df: pd.DataFrame = pd.DataFrame(dataset.y)
    class_counts: np.ndarray = df.sum().to_numpy()
    max_count: int = max(class_counts)
    class_imbalance_ratio: List = (class_counts / max_count).tolist()
    return class_imbalance_ratio
