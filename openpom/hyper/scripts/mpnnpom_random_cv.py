import os
import json
import torch
import tempfile
import numpy as np
from tqdm import tqdm
import deepchem as dc
from datetime import datetime
from openpom.models.mpnn_pom import MPNNPOMModel
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio
from openpom.hyper.configs.model_configs import MPNNPOMConfig
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

DATASET = 'openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv'
SMILES_FIELD = 'nonStereoSMILES'
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


def save_checkpoint(self,
                    max_checkpoints_to_keep: int = 5,
                    model_dir=None,
                    ckpt_name: str = 'best_checkpoint') -> None:
    """
        Save a checkpoint to disk.

        Parameters
        ----------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        model_dir: str, default None
            Model directory to save checkpoint to. If None, revert to self.model_dir
        """
    self._ensure_built()
    if model_dir is None:
        model_dir = self.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the checkpoint to a file.

    data = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self._pytorch_optimizer.state_dict(),
        'global_step': self._global_step
    }
    temp_file = os.path.join(model_dir, f'{ckpt_name}.pt')
    torch.save(data, temp_file)

    # Rename and delete older files.
    paths = [
        os.path.join(model_dir, f'{ckpt_name}%d.pt' % (i + 1))
        for i in range(max_checkpoints_to_keep)
    ]
    if os.path.exists(paths[-1]):
        os.remove(paths[-1])
    for i in reversed(range(max_checkpoints_to_keep - 1)):
        if os.path.exists(paths[i]):
            os.rename(paths[i], paths[i + 1])
    os.rename(temp_file, paths[0])


class CV:
    """
    K-FOLD CROSS VALIDATION for MPNNPOM model
    with custom stratification splitting
    """

    def __init__(self, model_builder, n_folds, device=None) -> None:
        self.model_builder = model_builder
        self.n_folds = n_folds
        self.device = device

    def _deepchem_splitter(self, dataset):
        randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
        return randomstratifiedsplitter.k_fold_split(dataset=dataset,
                                                     k=self.n_folds)

    def generate_folds(self, dataset, splitter='deepchem'):
        if splitter == 'deepchem':
            self.folds_list = self._deepchem_splitter(dataset)
        elif splitter == 'skmultilearn':
            raise NotImplementedError
        return self.folds_list

    def cross_validation(self,
                         model_params,
                         logdir=None,
                         max_epoch=100,
                         metric=None,
                         save_best_ckpt=False):
        logger.info("hyperparameters: %s" % str(model_params))
        all_folds_train_scores = []
        all_folds_val_scores = []
        if metric is None:
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        for fold_num, (train_dataset,
                       valid_dataset) in enumerate(self.folds_list):
            logger.info("Fitting model %d/%d folds" %
                        (fold_num + 1, self.n_folds))
            folder_name = f"fold_{fold_num + 1}_trial_count_{model_params['trial_count']}_{str(datetime.now())}"

            if logdir is not None:
                model_dir = os.path.join(logdir, folder_name)
                logger.info("model_dir is %s" % model_dir)
                try:
                    os.makedirs(model_dir)
                except OSError:
                    if not os.path.isdir(model_dir):
                        logger.info(
                            "Error creating model_dir, using tempfile directory"
                        )
                        model_dir = tempfile.mkdtemp()
            else:
                model_dir = tempfile.mkdtemp()

            model_params['model_dir'] = model_dir
            model_params['class_imbalance_ratio'] = get_class_imbalance_ratio(
                train_dataset)
            if self.device is not None:
                model_params['device_name'] = self.device
            model = self.model_builder(**model_params)

            best_train_score = 0  # train score for best validation
            best_val_score = 0
            error = ""
            try:
                for epoch in tqdm(range(1, max_epoch + 1)):
                    loss = model.fit(train_dataset,
                                     nb_epoch=1,
                                     max_checkpoints_to_keep=1,
                                     deterministic=False,
                                     restore=epoch > 1)

                    train_scores = model.evaluate(train_dataset,
                                                  [metric])['roc_auc_score']
                    valid_scores = model.evaluate(valid_dataset,
                                                  [metric])['roc_auc_score']
                    if valid_scores > best_val_score:
                        best_val_score = valid_scores
                        best_train_score = train_scores
                        if save_best_ckpt:
                            save_checkpoint(model, 1, None,
                                            f'best_ckpt_{fold_num}_')
                    logger.info(
                        f"epoch {epoch}/{max_epoch} ; loss = {loss}; train_scores = {train_scores}; test_scores = {valid_scores}"
                    )
            except Exception as e:
                error = f"Training error: {e}"

            all_folds_train_scores.append(best_train_score)
            all_folds_val_scores.append(best_val_score)

            try:
                os.remove(os.path.join(model_dir, 'checkpoint1.pt'))
            except:
                pass
            del model
            torch.cuda.empty_cache()

        mean_train_score = np.asarray(all_folds_train_scores).mean()
        mean_val_score = np.asarray(all_folds_val_scores).mean()
        logger.info("Results:")
        logger.info(f"hyperparameters: {str(model_params)}")
        logger.info(f"fold train scores: {all_folds_train_scores}")
        logger.info(f"fold validation scores: {all_folds_val_scores}")
        logger.info(f"mean train score: {mean_train_score}")
        logger.info(f"mean validation score: {mean_val_score}")
        return mean_train_score, mean_val_score, error


def random_search_cv(tasks=TASKS,
                     dataset=DATASET,
                     smiles_field=SMILES_FIELD,
                     n_folds=2,
                     n_trials=1,
                     logdir='./models',
                     max_epoch=10,
                     save_best_ckpt=False):
    # get dataset
    featurizer = GraphFeaturizer()
    smiles_field = smiles_field
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field=smiles_field,
                               featurizer=featurizer)
    input_file = dataset
    dataset = loader.create_dataset(inputs=[input_file])
    n_tasks = len(dataset.tasks)

    n_folds = n_folds
    n_trials = n_trials
    logdir = logdir
    max_epoch = max_epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Metric
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    model_builder = lambda **params: MPNNPOMModel(
        n_tasks=n_tasks,
        mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1,
        **params)

    cv = CV(model_builder=model_builder, n_folds=n_folds, device=device)
    cv.generate_folds(dataset=dataset, splitter='deepchem')

    try:
        file_name = f"{n_trials}_trials_params.json"
        file_path = os.path.join('./examples/trials', file_name)

        with open(file_path, 'r') as json_file:
            trials_dict = json.load(json_file)
    except:
        trials_dict, _ = MPNNPOMConfig.generate_hyperparams_random(
            n_trials=n_trials, dir='./examples/trials')

    logger.info("Starting random search crosss validation:")
    best_train_score = 0
    best_validation_score = 0
    best_hyperparams = {}
    all_scores = {}
    for trial_count, model_params in tqdm(trials_dict.items()):
        logger.info(f"{trial_count} starting:")
        model_params['trial_count'] = trial_count
        trial_start_time = datetime.now()
        mean_train_score, mean_val_score, error = cv.cross_validation(
            model_params=model_params,
            logdir=logdir,
            max_epoch=max_epoch,
            save_best_ckpt=save_best_ckpt,
            metric=metric)
        trial_end_time = datetime.now()

        all_scores[trial_count] = {
            'mean_train_score': mean_train_score,
            'mean_val_score': mean_val_score
        }

        current_date_time = str(datetime.now())
        log_file = os.path.join(
            logdir, f'results_{trial_count}_trial_{current_date_time}.txt')
        with open(log_file, 'w+') as f:
            f.write("Hyperparameters dictionary %s\n" % str(model_params))
            f.write("validation score %f\n" % mean_val_score)
            f.write("train_score: %f\n" % mean_train_score)
            f.write("trial_time: %s\n" %
                    str(trial_end_time - trial_start_time))
            f.write("error: %s\n" % error)

        if mean_val_score > best_validation_score:
            best_train_score = mean_train_score
            best_validation_score = mean_val_score
            best_hyperparams = model_params

    logger.info("Best hyperparameters: %s" % str(best_hyperparams))
    logger.info("best train_score: %f" % best_train_score)
    logger.info("best validation_score: %f" % best_validation_score)

    current_date_time = str(datetime.now())
    log_file = os.path.join(
        logdir, f'results_{n_trials}_trials_{current_date_time}.txt')

    with open(log_file, 'w+') as f:
        f.write("Best Hyperparameters dictionary %s\n" % str(best_hyperparams))
        f.write("Best validation score %f\n" % best_validation_score)
        f.write("Best train_score: %f\n" % best_train_score)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f",
                        "--n_folds",
                        default=2,
                        type=int,
                        help="Number of folds for cross-validation")
    parser.add_argument("-t",
                        "--n_trials",
                        default=2,
                        type=int,
                        help="Number of trials for random search cv")
    parser.add_argument("-d",
                        "--logdir",
                        default="./models",
                        help="Log directory to store results")
    parser.add_argument("-e",
                        "--max_epoch",
                        default=10,
                        type=int,
                        help="Number of epochs for each fold per trial")
    parser.add_argument("-c",
                        "--save_best_ckpt",
                        action="store_true",
                        help="Whether to save best checkpoints?")
    args = vars(parser.parse_args())

    n_folds = args['n_folds']
    n_trials = args['n_trials']
    logdir = args['logdir']
    max_epoch = args['max_epoch']
    save_best_ckpt = args['save_best_ckpt']
    random_search_cv(n_folds=n_folds,
                     n_trials=n_trials,
                     logdir=logdir,
                     max_epoch=max_epoch,
                     save_best_ckpt=save_best_ckpt)
