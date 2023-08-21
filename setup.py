from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='openpom',
    version='0.2.0',
    description='Open-source Principal Odor Map models for Olfaction',
    license='MIT',
    long_description="""Open Principal Odor Map

https://github.com/ARY2260/openpom

Replication of the Principal Odor Map paper by Lee et al (2022).
The model is implemented such that it integrates with DeepChem (https://github.com/deepchem/deepchem). 

Contributors:

Aryan Barsainyan: Code, data cleaning, model development

Ritesh Kumar: data cleaning, hyperparameter optimisation

Pinaki Saha: discussions and feedback

Michael Schmuker: Conceptualisation, project lead

References:

A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.

Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei,
Wesley W. Qian, Kelsie Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy,
Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, Alexander B. Wiltschko
bioRxiv 2022.09.01.504602; doi: (https://doi.org/10.1101/2022.09.01.504602)""",
    author='Aryan Amit Barsainyan',
    author_email='aryan.barsainyan@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
