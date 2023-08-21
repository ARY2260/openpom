# Open Principal Odor Map
Replication of the Principal Odor Map paper by Lee et al (2022) \[1\].
The model is implemented such that it integrates with [DeepChem](https://github.com/deepchem/deepchem ).

## Installation (Python 3.9 and above)
### PyPI based installation
1. `pip install openpom`

**openpom** requires [cuda verion of dgl libraries](https://www.dgl.ai/pages/start.html)<br>
For cuda 11.7, steps below can be followed:

2. `pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html`
3. `pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html`

## Contributors:
Aryan Barsainyan: Code, data cleaning, model development<br/>
Ritesh Kumar: data cleaning, hyperparameter optimisation<br/>
Pinaki Saha: discussions and feedback<br/>
Michael Schmuker: Conceptualisation, project lead<br/>

## References:
A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.<br/>
Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei, Wesley W. Qian, Kelsie Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy, Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, Alexander B. Wiltschko<br/>
bioRxiv 2022.09.01.504602; doi: [https://doi.org/10.1101/2022.09.01.504602](https://doi.org/10.1101/2022.09.01.504602)
