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

---

> [!IMPORTANT]
> **Automatic Engine Setup:** When importing RNACOREX, the package automatically checks whether `interactions.parquet` is present in the required `Engine/` directory. If the file is missing, it will be downloaded automatically.
