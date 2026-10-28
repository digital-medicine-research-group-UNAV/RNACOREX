(UNDER DEVELOPMENT)

This is the repository of the **RNA Co-Regulatory Network Explorer and Classifier (RNACOREX)** package, developed in the DATAI institute of Universidad de Navarra. **RNACOREX** is a Python package for building Bayesian Network based classification models using miRNA-mRNA post-transcriptional networks. It uses curated interaction databases and conditional mutual information for identifying sets of interactions and model them using Conditional Linear Gaussian Classifiers (CLGs).

## 🚀 Features

- Extracts structural and functional scores from RNA interactions.
- Identify post-transcriptional coregulation networks associated with different phenotypes.
- Build Bayesian network classifiers using these interaction sets.
- Display the post-transcriptional networks.

## New Functionalities

- Includes all kind of interactions among mRNAs, miRNAs and lncRNAs.
- Models nonlinear dependencies with SPBNs and SLGBNs.
- Allow multiclass classification and partial inference.
- Displays post-transcriptional networks and maximum probability networks.

## 📦 Installation

Until RNACOREX is available on PyPI, it can be installed locally from the root directory of the cloned repository:

```bash
python -m pip install -e .
```

> [!IMPORTANT]
> **Automatic Engine Setup:** When importing RNACOREX, the package automatically checks whether `interactions.parquet` is present in the required `Engine/` directory. If the file is missing, it will be downloaded automatically.

The engine can also be downloaded using the next command:

```bash
rnacorex.download()
```

Run the next command to check if the engines have been correctly added:

```bash
rnacorex.check_engines()
```

## ⚡ Quick Start

Run the `quickstart.py` script and `Quick_Start.ipynb` notebook in `quickstart` folder for easy applications of RNACOREX. Find data in `GPB/Data/` folder.

```python
from rnacorex import RNACOREX
import pandas as pd

# Load data

# Initialize model

rnacorex = RNACOREX(n_con=40)

# Fit model

rnacorex.fit(X_train, y_train)

# Load the network.

G = rnacorex.model_['G']

# Predict on new data

rnacorex.predict(X_test)

```

👉 **Note:** The input matrix must follow these ID formats:

- **mRNAs** should be identified using **Ensembl gene IDs** (e.g. `ENSG00000139618`) **without version numbers** (❌ `ENSG00000139618.12`).
- **miRNAs** should be named using **miRBase names in lowercase**, e.g. `hsa-mir-21` or `hsa-mir-125b-3p`.


## How to cite?
