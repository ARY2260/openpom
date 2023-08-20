from deepchem.models.optimizers import Optimizer
from deepchem.models.optimizers import Adam
from deepchem.models.optimizers import AdaGrad
from deepchem.models.optimizers import AdamW
from deepchem.models.optimizers import SparseAdam
from deepchem.models.optimizers import RMSProp
from deepchem.models.optimizers import GradientDescent
from deepchem.models.optimizers import KFAC


def get_optimizer(optimizer_name: str = 'adam') -> Optimizer:
    """
    Get deepchem optimizer object

    Parameters
    ---------
    optimizer_name: str
      optimizer name
      choices: [adam, adagrad, adamw, sparseadam, rmsprop, sgd, kfac]
      default: 'adam'

    Returns
    -------
    Optimizer
      Deepchem optimizer object
    """
    if optimizer_name == 'adam':
        return Adam()
    elif optimizer_name == 'adagrad':
        return AdaGrad()
    elif optimizer_name == 'adamw':
        return AdamW()
    elif optimizer_name == 'sparseadam':
        return SparseAdam()
    elif optimizer_name == 'rmsprop':
        return RMSProp()
    elif optimizer_name == 'sgd':
        return GradientDescent()
    elif optimizer_name == 'kfac':
        return KFAC()
    else:
        print("INVALID OPTIMISER NAME!, using ADAM optimizer by default")
        return Adam()
