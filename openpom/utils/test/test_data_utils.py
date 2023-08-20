from deepchem.data.data_loader import CSVLoader
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.utils.data_utils import get_class_imbalance_ratio


def test_class_imbalance_ratio():
    """
    Test get_class_imbalance_ratio utility on a test dataset
    """
    featurizer = GraphFeaturizer()
    smiles_field = 'smiles'
    loader = CSVLoader(tasks=['fruity', 'green', 'herbal', 'sweet', 'woody'],
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/utils/test/assets/test_dataset_sample_7.csv'
    dataset = loader.create_dataset(inputs=[input_file])
    class_imbalance_ratio = get_class_imbalance_ratio(dataset=dataset)
    assert class_imbalance_ratio == [1.0, 0.5, 0.5, 0.25, 0.5]
    assert class_imbalance_ratio[0] == 1.0  # max count of 'fruity'
