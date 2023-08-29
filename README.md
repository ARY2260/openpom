# Open Principal Odor Map
Replication of the Principal Odor Map paper by Lee et al (2022) \[1\].
The model is implemented such that it integrates with [DeepChem](https://github.com/deepchem/deepchem ).

## Installation (Python 3.9 and above)
### PyPI based installation
1. ```bash
   pip install openpom
   ```

**openpom** requires [cuda verion of dgl libraries](https://www.dgl.ai/pages/start.html)<br>
For cuda 11.7, steps below can be followed:

2. ```bash
   pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
   ```
3. ```bash
   pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
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
conda create -n open_pom python=3.9
conda activate open_pom
```

3. Install openpom

```bash
pip install .
```

or (for developing)
```bash
python setup.py develop
```
4. Install DGL cuda libs
```bash
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html

pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Getting started
Example notebooks for model training and finetuning are available [here](https://github.com/ARY2260/openpom/tree/main/examples).

## Contributors:
**Aryan Barsainyan**, National Institute of Technology, Karnataka, India: code, data cleaning, model development<br/>
**Ritesh Kumar**, CSIR-CSIO, Chandigarh, India: data cleaning, hyperparameter optimisation<br/>
**Pinaki Saha**, University of Hertfordshire, UK: discussions and feedback<br/>
**Michael Schmuker**, University of Hertfordshire, UK: conceptualisation, project lead<br/>

## References:
\[1\] A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.<br/>
Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei, Wesley W. Qian, Kelsie Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy, Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, Alexander B. Wiltschko<br/>
bioRxiv 2022.09.01.504602; doi: [https://doi.org/10.1101/2022.09.01.504602](https://doi.org/10.1101/2022.09.01.504602)
