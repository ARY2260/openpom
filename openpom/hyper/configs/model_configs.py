# yapf: disable
import numpy as np
from typing import Dict
from openpom.hyper.configs.base_config import Config


class MPNNPOMConfig(Config):
    """
    Hyperparameter search space for MPNNPOMModel

    Note:
        MPNNPOMConfig.generate_hyperparams_random() method
        might take a lot of memory during computation of
        combinations, so tweak the parameter space accordingly.
    """
    PARAMS_DICT: Dict = {
        'batch_size': [128, 256],
        'node_out_feats': list(range(50, 300, 50)),
        'edge_hidden_feats': list(range(50, 200, 25)),
        'edge_out_feats': list(range(50, 300, 50)),
        'num_step_message_passing': list(range(1, 6, 1)),
        'mpnn_residual': [True, False],
        'message_aggregator_type': ['sum', 'mean'],
        'readout_type': ['set2set', 'global_sum_pooling'],
        'num_step_set2set': [2, 3, 4, 5],
        'num_layer_set2set': [1, 2, 3],
        'ffn_hidden_list': [
                            [392],
                            [512],
                            [392, 392],
                            [512, 512],
                            [392, 392, 392],
                            [512, 512, 512],
                            ],
        'ffn_embeddings': [256],
        'ffn_dropout_p': np.linspace(0.05, 0.5, num=5).tolist(),
        'ffn_dropout_at_input_no_act': [True, False],
        'weight_decay': [0.001, 0.0001, 1e-05, 1e-06],
        'learning_rate': [0.0005, 0.001, 0.005, 0.01],
        'self_loop': [True, False],
        }
