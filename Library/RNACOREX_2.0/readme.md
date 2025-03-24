# RNACOREX 2.0

RNACOREX 2.0 is a package that extracts coregulation networks associated to a specific phenotype. The networks are composed of interactions between mRNAs, miRNAs and lncRNAs. The package uses a hybrid approach, combining expert information from well-known databases such as TargetScan [1], DIANA [2] and miRTarBase [3] for mRNA-miRNA interactions, GeneRIF [4] for mRNA-mRNA interactions and NPInter [5], LNCRNASNP [6] and LNCBook [7] for lncRNA related interactions, with an empirical analysis of expression data. The package develops Conditional Linear Gaussian Classifiers (CLGs) in order to identify the most relevant set of interactions associated to specific pathologies and classify new samples.

## Requirements

Please, read library instructions before using the library.

RNACOREX implements the next libraries and versions. Correct operation of the package cannot be ensured with other versions.

`Python` 3.10.15 +

`gtfparse` 2.5.0 + (install using pip)

`matplotlib` 3.8.4 +

`networkx` 3.1 +

`numpy` 1.26.4 +

`pandas` 2.1.4 +

`scipy` 1.13.1 +

`scikit-learn` 1.4.2 +

`tqdm` 4.65.0 +

# Quick Start

Firstly, we will import the required modules.

```python

import pandas as pd

from sklearn.model_selection import train_test_split

import miRNetClassifier

```
