import os
import numpy as np
from openpom.hyper.configs.base_config import Config


def test_config():
    """
    Test for Config base class
    """
    np.random.seed(0)
    Config.PARAMS_DICT = {
        "param1": ["p1v1", "p1v2"],
        "param2": ["p2v1", "p2v2", "p2v3"]
    }
    n_trials = 2
    param_dict, path = Config.generate_hyperparams_random(
        n_trials=n_trials, dir="openpom/hyper/test/assets")
    assert isinstance(param_dict, dict)
    assert isinstance(path, str)
    assert len(param_dict.keys()) == n_trials
    for _, params in param_dict.items():
        assert "param1" in params
        assert "param2" in params

    # remove saved file
    os.remove(path)
