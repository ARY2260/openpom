import tempfile
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Iterator
from deepchem.data.datasets import DiskDataset, NumpyDataset
from skmultilearn.model_selection import IterativeStratification
from deepchem.splits import Splitter


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
    if not isinstance(dataset, DiskDataset) and not isinstance(dataset, NumpyDataset):
        raise Exception("The dataset should be a deepchem DiskDataset or NumpyDataset")
    df: pd.DataFrame = pd.DataFrame(dataset.y)
    class_counts: np.ndarray = df.sum().to_numpy()
    max_count: int = max(class_counts)
    class_imbalance_ratio: List = (class_counts / max_count).tolist()
    return class_imbalance_ratio


class IterativeStratifiedSplitter(Splitter):
    """
    Iteratively stratify a multi-label data set into folds/splits.

    Construct an iterative stratifier that splits the dataset
    trying to maintain balanced representation with respect to
    order-th label combinations.

    Available splits:
        - train_valid_test_split()
        - train_test_split()

    Note:
        Requires `skmultilearn` library to be installed.
    """

    def __init__(self, order: int = 2) -> None:
        """
        Parameters
        ---------
        order: int
            order for iterative stratification (default: 2)
        """
        self.order: int = order

    def split(
        self,
        dataset: DiskDataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return indices for iterative stratified split

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to be split.
        seed: int, optional (default None)
            Random seed to use.
        frac_train: float, optional (default 0.8)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
            The fraction of data to be used for the test split.
        log_every_n: int, optional (default None)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple `(train_indices, valid_indices, test_indices)`
            for the various splits.
        """
        X1: pd.DataFrame
        y1: pd.DataFrame
        X1, y1 = pd.DataFrame(dataset.X), pd.DataFrame(dataset.y)
        stratifier1: IterativeStratification = IterativeStratification(
            n_splits=2,
            order=self.order,
            sample_distribution_per_fold=[frac_test + frac_valid, frac_train],
            # shuffle=True,
            random_state=seed,
        )

        train_indices: np.ndarray
        other_indices: np.ndarray
        train_indices, other_indices = next(stratifier1.split(X1, y1))

        temp_dir: str = tempfile.mkdtemp()
        other_dataset: DiskDataset = dataset.select(other_indices.tolist(),
                                                    temp_dir)

        X2: pd.DataFrame
        y2: pd.DataFrame
        X2, y2 = pd.DataFrame(other_dataset.X), pd.DataFrame(other_dataset.y)
        new_split_ratio: float = round(frac_test / (frac_test + frac_valid), 2)
        stratifier2: IterativeStratification = IterativeStratification(
            n_splits=2,
            order=self.order,
            sample_distribution_per_fold=[
                new_split_ratio, 1 - new_split_ratio
            ],
            random_state=seed,
        )

        valid_indices: np.ndarray
        test_indices: np.ndarray
        valid_indices, test_indices = next(stratifier2.split(X2, y2))
        return train_indices, valid_indices, test_indices

    def k_fold_split(
        self,
        dataset: DiskDataset,
        k: int,
        directories: Optional[List[str]] = None
    ) -> List[Tuple[DiskDataset, DiskDataset]]:
        """
        Parameters
        ----------
        dataset: DiskDataset
            DiskDataset to do a k-fold split
        k: int
            Number of folds to split `DiskDataset` into. (k>1)
        directories: List[str], optional (default None)
            List of length 2*k filepaths to save the result disk-datasets.

        Returns
        -------
        List[Tuple[DiskDataset, DiskDataset]]
            List of length k tuples of (train, cv)
            where `train` and `cv` are both `DiskDataset`.
        """
        assert k != 1
        if directories is None:
            directories = [tempfile.mkdtemp() for _ in range(2 * k)]
        else:
            assert len(directories) == 2 * k

        X: pd.DataFrame
        y: pd.DataFrame
        X, y = pd.DataFrame(dataset.X), pd.DataFrame(dataset.y)
        stratifier: IterativeStratification = IterativeStratification(
            n_splits=k,
            order=self.order,
        )

        train_datasets: List = []
        cv_datasets: List = []
        split_gen: Iterator = stratifier.split(X, y)
        for fold in range(k):
            train_dir, cv_dir = directories[2 * fold], directories[2 * fold +
                                                                   1]
            train_indices: np.ndarray
            cv_indices: np.ndarray
            train_indices, cv_indices = next(split_gen)
            train_dataset: DiskDataset = dataset.select(
                train_indices.tolist(), train_dir)
            cv_dataset: DiskDataset = dataset.select(cv_indices.tolist(),
                                                     cv_dir)
            train_datasets.append(train_dataset)
            cv_datasets.append(cv_dataset)
        return list(zip(train_datasets, cv_datasets))
