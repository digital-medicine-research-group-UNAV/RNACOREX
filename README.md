This is the repository of the **RNA Co-Regulatory Network Explorer and Classifier (RNACOREX)** package, developed in the DATAI institute of Universidad de Navarra. **RNACOREX** is a Python package for building Bayesian Network based classification models using miRNA-mRNA post-transcriptional networks. It uses curated interaction databases and conditional mutual information for identifying sets of interactions and model them using Conditional Linear Gaussian Classifiers (CLGs).

- PyPI: [RNACOREX on PyPI](https://pypi.org/project/RNACOREX/).

- Zenodo: [RNACOREX on Zenodo](https://zenodo.org/records/17368843).

Using `Python <= 3.9` and `numpy <= 2` is strongly recommended.

---

## 🚀 Features

- Extracts structural and functional scores from miRNA-mRNA interactions.
- Identify sets of interactions associated with different phenotypes.
- Build CLG classifiers using these interaction sets.
- Display the post-transcriptional networks.

---

## 📦 Installation

Installation in a Python virtual environment is required. It is highly recommended to run it in a **conda environment**. Install with:

```bash
pip install rnacorex
```

**Important:** Next engines must be placed in their path `rnacorex\engines` **before** running the package. 

- `DIANA_targets.txt`
- `Tarbase_v9.tsv`
- `Targetscan_targets.txt`
- `MTB_targets_25.txt`
- `gencode.v47.basic.annotation.gtf`

Engines can be downloaded using the next command:

```bash
rnacorex.download()
```

Or manually: [DOWNLOAD](https://tinyurl.com/RNACOREX)

Run the next command to check if the engines have been correctly added:

```bash
rnacorex.check_engines()
```

**Important:** For displaying networks,`pygraphviz` must be installed separately using conda:

```bash
conda install -c conda-forge pygraphviz
```

---

## ⚡ Quick Start

Run the `Quick_Start.ipynb` notebook in `quickstart` folder for an easy application of RNACOREX.

Find data in `data` folder.

👉 **Note:** The input matrix must follow these ID formats:

- **mRNAs** should be identified using **Ensembl gene IDs** (e.g. `ENSG00000139618`) **without version numbers** (❌ `ENSG00000139618.12`).
- **miRNAs** should be named using **miRBase names in lowercase**, e.g. `hsa-mir-21` or `hsa-mir-125b-3p`.


## How to cite?

```
RNACOREX - RNA coregulatory network explorer and classifier
Oviedo-Madrid A, González-Gomariz J, Armañanzas R (2025)
RNACOREX - RNA coregulatory network explorer and classifier.
PLOS Computational Biology 21(11): e1013660.
https://doi.org/10.1371/journal.pcbi.1013660
```
