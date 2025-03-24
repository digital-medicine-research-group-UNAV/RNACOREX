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

Firstly, we will import the RNACOREX class and the required modules.

```python

import RNACOREX

import pandas as pd

from sklearn.model_selection import train_test_split

```

We charge the data and do the train-test splitting.

```python

data = pd.read_csv('data_complete.csv', index_col = 0)

X = data.drop('Class', axis = 1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

A RNACOREX model is initialized using the `n_con` and `precision` parameters. `n_con` defines the number of interactions used for constructing the network, which can be a single value, a specific set of values or a full range of values. `precision` stands for the precision of the conditional mutual information estimation, increasing complexity of calculations by uding bigger precisions. 

```python

# Constructing a model with k = 150.

rnacorex = RNACOREX.RNACOREX(n_con = 150, precision=10)

# Constructing three models with k = 100, k = 150 and k = 200.

rnacorex2 = RNACOREX.RNACOREX(n_con = [100, 150, 200], precision=10)

# Constructing several models with k ranging from 100 to 150.

rnacorex3 = RNACOREX.RNACOREX(n_con = (100, 150), precision=10)

```

By default, are fitted using all kind of interactions. If only specific elements are required, these can be selected from the model definition.

```python

rnacorex4 = RNACOREX.RNACOREX(X_train, y_train, mrna = True, mirna = True, lncrna = False)

```

Using train and test sets the model can be easily fitted.

```python

rnacorex.fit(X_train, y_train)

```

```python

rl = rnacorex.predict(X_test)

```

