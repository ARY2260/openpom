# Open Principal Odor Map
Replication of the Principal Odor Map paper by Brian K. Lee et al. (2023) \[1\].
The model is implemented such that it integrates with [DeepChem](https://github.com/deepchem/deepchem ).

## Benchmarks
| Model    | Data      | Type  | ROC-AUC Score |
| :------------: |   :---:       |   :---:       | :--------: |
| [MPNNPOMModel](https://github.com/ARY2260/openpom/blob/74e964eb5b1086badcb3e3ba47df3528259d7000/openpom/models/mpnn_pom.py)        |  [curated_GS_LF_merged_4983.csv](https://github.com/ARY2260/openpom/blob/74e964eb5b1086badcb3e3ba47df3528259d7000/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv)       | 5-Fold CV with ensemble of 10 models per fold   | 0.8872

## Installation (Python 3.10)
### PyPI based installation
1. Install latest Deepchem (nightly) version:

```bash
pip install --pre deepchem
```

2. Install torch v2.4.0 - CUDA 12.4 and dgl (for respective torch version)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

3. Install openpom

```bash
pip install openpom
```

### Github fork based installation
1. Fork the [OpenPOM](https://github.com/ARY2260/openpom) repository
and clone the forked repository

```bash
git clone https://github.com/YOUR-USERNAME/openpom.git
cd openpom
```

2. Setup conda environment
```bash
conda create -n open_pom python=3.10
conda activate open_pom
```

3. Install required dependencies and openpom

```bash
pip install --pre deepchem
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install pyrfume dgllife pytest ipykernel scikit-multilearn
pip install .
```

## Getting started
Example notebooks for model training and finetuning are available [here](https://github.com/ARY2260/openpom/tree/main/examples).

### Odor Prediction Demo: `predict_odors.py`

A ready-to-use demo GUI for predicting odor qualities from molecular SMILES strings is provided in [`predict_odors.py`](./predict_odors.py).
This Python script allows you to enter a SMILES string and obtain model-based odor predictions, visualized as a barplot of the top-N scoring odor terms.

#### Features

- **Ensemble MPNN Models:** Uses a openpom 10-model ensemble trained on the Principal Odor Map (POM) dataset.
- **Graphical User Interface:** Enter SMILES, select the number of top odors to display, and visualize results interactively.
- **RDKit-based Structure Rendering:** Shows the molecular structure (if RDKit is installed).
- **One-click Results Export:** Copy top-N predictions as TXT or JSON for downstream analysis.

#### Running the Demo

Assuming you have installed the required dependencies and downloaded or trained the ensemble models to `./models/ensemble_models`, launch the GUI with:
```bash
python predict_odors.py
```
The default window will prompt you to input a SMILES string. Press **Predict Odors** to view results.

- The left side of the interface displays the input entry and buttons.
- The right side shows the predicted odor probabilities as a bar chart for the top N terms.
- Results can be easily copied as text or JSON for practical use.

> **Note:** If you do not have GPU support (CUDA), ensure the models are set to load on CPU. You may have to edit the `device_name` parameter in the script accordingly.

#### Example Usage

1. **Input a SMILES string:**  
   For example, `CCO` (ethanol).
2. **Choose Top N:**  
   Specify how many of the highest-scoring odors to visualize.
3. **Press "Predict Odors":**  
   The barplot will show the predicted intensity for each top odor term.
4. **Copy Results:**  
   Use **Copy Results (TXT/JSON)** to export the predictions for further processing.


## Contributors:
**Aryan Amit Barsainyan**, National Institute of Technology Karnataka, India: code, data cleaning, model development<br/>
**Ritesh Kumar**, CSIR-CSIO, Chandigarh, India: data cleaning, hyperparameter optimisation<br/>
**Pinaki Saha**, University of Hertfordshire, UK: discussions and feedback<br/>
**Michael Schmuker**, University of Hertfordshire, UK: conceptualisation, project lead<br/>

## References:
\[1\] A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.<br/>

Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei, Wesley W. Qian, Kelsie A. Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy, Jacob Yasonik, Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, Alexander B. Wiltschko<br/>

Science381,999-1006(2023).DOI: [10.1126/science.ade4401](https://doi.org/10.1126/science.ade4401) <br/>
bioRxiv 2022.09.01.504602; doi: [https://doi.org/10.1101/2022.09.01.504602](https://doi.org/10.1101/2022.09.01.504602)

## Citing OpenPOM:
```
@misc{OpenPOM,
  author={Aryan Amit Barsainyan and Ritesh Kumar and Pinaki Saha and Michael Schmuker},
  title={OpenPOM - Open Principal Odor Map},
  year={2023},
  note={\url{https://github.com/BioMachineLearning/openpom}},
}
```
