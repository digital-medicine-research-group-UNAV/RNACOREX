# RNACOREX

**RNACOREX** is a Python package for building Bayesian Network based classification models using miRNA-mRNA post-transcriptional networks. It uses curated interaction databases and conditional mutual information for identifying sets of interactions and model them using Conditional Linear Gaussian Classifiers (CLGs).

Repository: [RNACOREX on GitHub](https://github.com/digital-medicine-research-group-UNAV/RNACOREX)

---

## 🚀 Features

- Extracts structural and functional scores from miRNA-mRNA interactions.
- Identify sets of interactions associated with different phenotypes.
- Build CLG classifiers using these interaction sets.
- Display the post-transcriptional networks.


## 📦 Installation

Installation in a Python virtual environment is required. It is highly recommended to run it in a **conda environment**.

Install with:

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

Alternatively they can be manually downloaded from: `https://tinyurl.com/RNACOREX`

Run the next command to check if the engines have been correctly added:

```bash
rnacorex.check_engines()
```

**Important:** For displaying networks,`pygraphviz` must be installed separately using conda:

```bash
conda install -c conda-forge pygraphviz
```



