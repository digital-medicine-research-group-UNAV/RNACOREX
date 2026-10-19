# 🧬 Reproducibility — Paper Experiments

This directory contains the necessary code to **reproduce the experiments** described in the associated publication.  
Below are detailed instructions to run the complete experimental workflow.

---

## 📂 Repository Structure

```text
├── preprocessing/        # Scripts for TCGA data preprocessing
├── other_scripts/        # Additional scripts: figures, benchmarking, etc.
├── main_plos.py          # Main script to reproduce the experiments
└── README.md             # This file
```

---

## 🔧 1. Data Preprocessing

The scripts in the `preprocessing/` folder must be executed **before reproducing the main experiments**.

These scripts:
- Use **raw TCGA data** (The Cancer Genome Atlas).
- Perform the **preprocessing and cleaning** steps to generate the input datasets used by the models.

> ⚠️ **Important:**  
> The raw TCGA data **are not included in this repository** due to GitHub size limitations.  
> They can be manually accessed in the next link:

🔗 [UCSC Xena](https://xenabrowser.net/datapages/?hub=https://gdc.xenahubs.net:443)

---

## 📊 2. Required Data

The following TCGA datasets are required:

- **STAR counts** (gene expression)
- **miRNA expression**
- **Survival data**

These data must be obtained for the following 13 TCGA cohorts:

1. TCGA-BRCA
2. TCGA-COAD
3. TCGA-HNSC
4. TCGA-KIRC
5. TCGA-LAML
6. TCGA-LGG
7. TCGA-LIHC
8. TCGA-LUAD  
9. TCGA-LUSC  
10. TCGA-SKCM  
11. TCGA-SARC     
12. TCGA-STAD  
13. TCGA-UCEC 

---

## 🧠 3. Running the Main Experiments

The script `main_plos.py` uses the preprocessed datasets to:

- Run the **main experiments** described in the paper.  
- Compute the **performance metrics** reported in the article.

Example execution:

```bash
python main_plos.py
```

## 🧩 4. Additional Scripts and Results

The `other_scripts/` folder contains additional utilities:

- Scripts to reconstruct figures and images from the paper.

- Scripts to run the benchmarking with CGBayesNets (RNACOREX implementation).

- Additional helper scripts used in the analysis.

Data, figures and results are expected to be stored in three folders: `data\`, `figures\` and `results\`.

## 📘 Recommended Execution Order

It is recommended to execute the scripts in the following order:

1. `preprocessing/`

2. `main_plos.py`

3. `other_scripts/` (optional for complementary results)






















