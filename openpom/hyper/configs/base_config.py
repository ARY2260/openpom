import os
import json
import itertools
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class Config:
    """
    Abstract base class for model configurations.
    It defines hyperparameter search space
    for a model.

    Supported methods:
    - generate_hyperparams_random
        Generate hyperparameter combinations
        for random search.
    """
    PARAMS_DICT: Dict[str, List] = {
        'sample_param': ['sample_value1', 'sample_value2']
    }

    @classmethod
    def generate_hyperparams_random(
            cls,
            n_trials: int = 1,
            dir: Optional[str] = None) -> Tuple[Dict, str]:
        """
        Generate hyperparameter combinations for random trials.

        Parameters
        ----------
        n_trials: int
            Number of trials for random search.
        dir: Optional[str]
            Directory path to save json file of generated
            combinations.

        Returns
        -------
        trials_dict: Dict
            Dictionary of combinations generated for trials.
        file_path: str
            File path of saved json file containing generated
            combinations.
        """
        hyperparameter_combs: List[Dict[
            str, Any]] = cls._generate_random_hyperparam_values(n=n_trials)

        trials_dict: Dict = {}
        for count, params in enumerate(hyperparameter_combs):
            trials_dict[f'trial_{count+1}'] = params

        file_name: str = f"{n_trials}_trials_params.json"
        if dir is None:
            cwd: str = os.getcwd()
            file_path: str = os.path.join(cwd, file_name)
        else:
            file_path = os.path.join(dir, file_name)

        with open(file_path, "w") as json_file:
            json.dump(trials_dict, json_file, indent=4)

        return trials_dict, file_path

    @classmethod
    def _generate_random_hyperparam_values(cls,
                                           n: int) -> List[Dict[str, Any]]:
        """
        Generates `n` random hyperparameter combinations
        of hyperparameter values

        Parameters
        ----------
        n: int
            Number of random combinations to generate

        Returns
        -------
        params_subset: List[Dict[str, Any]]
            list of hyperparameter combinations
        """
        hyperparam_keys: List
        hyperparam_values: List
        hyperparam_keys, hyperparam_values = [], []
        for key, values in cls.PARAMS_DICT.items():
            if callable(values):
                # If callable, sample it for a maximum n times
                values = [values() for i in range(n)]
            hyperparam_keys.append(key)
            hyperparam_values.append(values)

        hyperparam_combs: List = []
        for iterable_hyperparam_comb in itertools.product(*hyperparam_values):
            hyperparam_comb: List = list(iterable_hyperparam_comb)
            hyperparam_combs.append(hyperparam_comb)
        indices: np.ndarray = np.random.permutation(len(hyperparam_combs))[:n]

        params_subset: List = []
        for index in indices:
            param: Dict = {}
            for key, hyperparam_value in zip(hyperparam_keys,
                                             hyperparam_combs[index]):
                param[key] = hyperparam_value
            params_subset.append(param)
        return params_subset
