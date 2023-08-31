from deepchem.data.data_loader import CSVLoader
from deepchem.data.datasets import DiskDataset
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.utils.data_utils import get_class_imbalance_ratio
from openpom.utils.data_utils import IterativeStratifiedSplitter

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus',
    'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked',
    'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry',
    'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh',
    'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium', 'grape',
    'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut', 'herbal',
    'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender',
    'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal',
    'melon', 'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk',
    'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion', 'orange',
    'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine',
    'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish',
    'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood', 'savory',
    'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry',
    'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato',
    'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy',
    'weedy', 'winey', 'woody'
]


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


def test_IS_split_train_test():
    """
    Test train-test split of IterativeStratifiedSplitter
    """
    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = CSVLoader(tasks=TASKS,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/utils/test/assets/large_test_dataset.csv'
    dataset = loader.create_dataset(inputs=[input_file])
    splitter = IterativeStratifiedSplitter(order=2)

    train, test = splitter.train_test_split(dataset, frac_train=0.7)
    assert isinstance(train, DiskDataset)
    assert isinstance(test, DiskDataset)
    assert round(len(train) / (len(train) + len(test)), 1) == 0.7


def test_IS_split_train_valid_test():
    """
    Test train-valid-test split of IterativeStratifiedSplitter
    """
    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = CSVLoader(tasks=TASKS,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/utils/test/assets/large_test_dataset.csv'
    dataset = loader.create_dataset(inputs=[input_file])
    splitter = IterativeStratifiedSplitter(order=2)

    train, valid, test = splitter.train_valid_test_split(dataset,
                                                         frac_train=0.7,
                                                         frac_valid=0.2,
                                                         frac_test=0.1)
    assert isinstance(train, DiskDataset)
    assert isinstance(valid, DiskDataset)
    assert isinstance(test, DiskDataset)
    assert round(len(train) / (len(train) + len(valid) + len(test)), 1) == 0.7
    assert round(len(valid) / (len(train) + len(valid) + len(test)), 1) == 0.2


def test_IS_split_kfold():
    """
    Test kfold split of IterativeStratifiedSplitter
    """
    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = CSVLoader(tasks=TASKS,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    input_file = \
        'openpom/utils/test/assets/large_test_dataset.csv'
    dataset = loader.create_dataset(inputs=[input_file])
    splitter = IterativeStratifiedSplitter(order=2)

    folds_list = splitter.k_fold_split(dataset, k=5)
    assert len(folds_list) == 5
    assert len(folds_list[0]) == 2
    assert isinstance(folds_list[0][0], DiskDataset)
    assert isinstance(folds_list[0][1], DiskDataset)
