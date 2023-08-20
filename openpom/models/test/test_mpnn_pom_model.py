import torch
import tempfile
import numpy as np
import deepchem as dc
from deepchem.data.data_loader import CSVLoader
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOM, MPNNPOMModel
from openpom.utils.data_utils import get_class_imbalance_ratio


def test_mpnnpom_model_classification():
    """
    Test MPNNPOMModel class for classification
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # load sample dataset
    featurizer = GraphFeaturizer()
    smiles_field = 'smiles'
    tasks = ['fruity', 'green', 'herbal', 'sweet', 'woody']
    loader = CSVLoader(tasks=tasks,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/models/test/assets/test_dataset_sample_7.csv'
    dataset = loader.create_dataset(inputs=[input_file])
    class_imbalance_ratio = get_class_imbalance_ratio(dataset=dataset)

    model = MPNNPOMModel(n_tasks=len(tasks),
                         batch_size=2,
                         class_imbalance_ratio=class_imbalance_ratio,
                         mode="classification",
                         n_classes=1,
                         device_name=device)

    assert isinstance(model.model, MPNNPOM)
    # overfit test
    model.fit(dataset, nb_epoch=50)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = model.evaluate(dataset, [metric])
    assert scores['roc_auc_score'] > 0.9


def test_mpnnpom_model_regression():
    """
    Test MPNNPOMModel class for regression
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # load sample dataset
    featurizer = GraphFeaturizer()
    smiles_field = 'smiles'
    tasks = ['y']
    loader = CSVLoader(tasks=tasks,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/models/test/assets/test_regression_sample.csv'
    dataset = loader.create_dataset(inputs=[input_file])

    model = MPNNPOMModel(
        n_tasks=len(tasks),
        batch_size=2,
        mode="regression",
    )

    assert isinstance(model.model, MPNNPOM)
    # overfit test
    model.fit(dataset, nb_epoch=100)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.5


def test_mpnnpom_model_reload():
    """
    Test MPNNPOMModel class for model reload
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # load sample dataset
    featurizer = GraphFeaturizer()
    smiles_field = 'smiles'
    tasks = ['y']
    loader = CSVLoader(tasks=tasks,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/models/test/assets/test_regression_sample.csv'
    dataset = loader.create_dataset(inputs=[input_file])

    # initialize the model
    model_dir = tempfile.mkdtemp()
    model = MPNNPOMModel(n_tasks=len(tasks),
                         batch_size=2,
                         mode="regression",
                         model_dir=model_dir)

    # fit the model
    model.fit(dataset, nb_epoch=10)

    # reload the model
    reloaded_model = MPNNPOMModel(n_tasks=len(tasks),
                                  batch_size=2,
                                  mode="regression",
                                  model_dir=model_dir)
    reloaded_model.restore()

    orig_predict = model.predict(dataset)
    reloaded_predict = reloaded_model.predict(dataset)
    assert np.all(orig_predict == reloaded_predict)
