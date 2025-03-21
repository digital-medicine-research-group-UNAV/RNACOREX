### PLOS CODE Instructions

* Place the required datasets in `TCGA Raw Data` (See library instructions)".

"(09/12/2024) The mRNA expression data format in UCSC Xena has been changed from HTSeq_Counts to STAR_Counts, please take into account before running the article code.

* `TCGA Raw Data` contains the original data extracted from GDC without preprocessing.

* `DEG R` contains R scripts for DEG. Datasets for DEG and differentially expressed genes are saved here. 

* For the Differential Gene Expression (DGE) developed in the article code the next requirements have to be taken into account:

  - `R` 4.3.0 +

  - `edgeR` 4.0.16 +

* In `Clean Data` the final database is saved.

* `PLOS Results` contains the main outputs of paper code.


### EXECUTION PROCESS

* `plos_code.py` can be directly executed without running all the preprocessing as the Clean Data is already saved in 'Clean Data' folder.

For full process execution:

1) `preprocessing_1.py` runs the first part of the preprocessing process.

2) `DEG R/edgeR_DEG_bbdd.R` scripts executes the DEG of the specific 'data_deg_bbdd.csv' database. 'DEG R/master.R' executes DEG for all databases.

3) `preprocessing_2.py` runs the second part of the preprocessing process using the DEG genes. Generates the final database and saves it in 'Clean Data' folder. It is executed for all the databases so it will not work if the DEG scripts are not executed in every database.

4) `plos_code.py` executes the script for building networks and extract metrics using RNACOREX. Outputs are saved in 'PLOS Results'.
