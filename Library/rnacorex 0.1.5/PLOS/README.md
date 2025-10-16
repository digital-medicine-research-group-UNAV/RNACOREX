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

---

## 🔧 1. Data Preprocessing

The scripts in the `preprocessing/` folder must be executed **before reproducing the main experiments**.

These scripts:
- Use **raw TCGA data** (The Cancer Genome Atlas).
- Perform the **preprocessing and cleaning** steps to generate the input datasets used by the models.

> ⚠️ **Important:**  
> The raw TCGA data **are not included in this repository** due to GitHub size limitations.  
> They must be downloaded manually from the official GDC portal:

🔗 [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)

---

## 📊 2. Required Data

The following TCGA datasets are required:

- **STAR counts** (gene expression)
- **miRNA expression**
- **Survival data**

These data must be obtained for the following 13 TCGA cohorts:

1. TCGA-BRCA  
2. TCGA-LUAD  
3. TCGA-LUSC  
4. TCGA-COAD  
5. TCGA-READ  
6. TCGA-KIRC  
7. TCGA-KIRP  
8. TCGA-LGG  
9. TCGA-GBM  
10. TCGA-STAD  
11. TCGA-THCA  
12. TCGA-HNSC  
13. TCGA-OV  

> Once downloaded, place the raw data in the directories expected by the scripts in `preprocessing/`.

---

## 🧠 3. Running the Main Experiments

The script `main_plos.py` uses the preprocessed datasets to:

- Run the **main experiments** described in the paper.  
- Compute the **performance metrics** reported in the article.

Example execution:

```bash
python main_plos.py
This script will generate the experimental results in the output folders specified in the code (check comments for paths and parameter settings).

🧩 4. Additional Scripts and Results
The other_scripts/ folder contains additional utilities:

Scripts to reconstruct figures and images from the paper.

Scripts to run the benchmarking with CGBayesNets (Python implementation).

Additional helper scripts used in the analysis.

📘 Recommended Execution Order
It is recommended to execute the scripts in the following order:

preprocessing/

main_plos.py

other_scripts/ (optional — for complementary results)

🧾 Citation
If you use this code or data in your research, please cite the associated paper












