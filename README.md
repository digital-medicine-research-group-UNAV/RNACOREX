(UNDER DEVELOPMENT)

This is the repository of the **RNA Co-Regulatory Network Explorer and Classifier (RNACOREX)** package, developed in the DATAI institute of Universidad de Navarra. **RNACOREX** is a Python package for building Bayesian Network based classification models using miRNA-mRNA post-transcriptional networks. It uses curated interaction databases and conditional mutual information for identifying sets of interactions and model them using Conditional Linear Gaussian Classifiers (CLGs).

> [!IMPORTANT]
> **Automatic Engine Setup:** When importing RNACOREX, the package automatically checks whether `interactions.parquet` is present in the required `Engine/` directory. If the file is missing, it will be downloaded automatically. Make sure you have an active internet connection and that the file is accessible from the download source.
