# RNACOREX

RNACOREX is a package that extracts the post-transcriptional coregulation network associated to a specific phenotype. The package uses a hybrid approach, combining structural information from well-known miRNA-mRNA interaction databases such as TargetScan [1], DIANA [2], miRTarBase [3] and TarBase [CITE] with an empirical analysis of expression data for developing Conditional Linear Gaussian Classifiers (CLGs) to identify the most relevant miRNA-mRNA interactions and classify new samples.

## Requirements

Please, read library instructions before using the library.

RNACOREX implements the next libraries and versions. Correct operation of the package cannot be ensured with other versions.

`Python` 3.9 +

`gtfparse` 2.5.0 + 

`matplotlib` 3.8.4 +

`networkx` 3.1 +

`numpy` 1.26.4 +

`pandas` 2.1.4 +

`pygraphviz` 1.9 +

`scipy` 1.13.1 +

`scikit-learn` 1.4.2 +

`tqdm` 4.65.0 +

## Library Instructions

Due to GitHub file size limit some files required by the packages have not been uploaded. These files should be downloaded and placed in their location manually by the user before running the library.

### Structural Engine Files

The Structural Engine (TargetScan, DIANA, miRTarBase & TarBase) and the Gencode files are compulsory to run the library. It will not work without them.

Download from: https://tinyurl.com/StructuralEngine.

Files should be downloaded and placed with their default names in the next paths: `Library/RNACOREX/StructuralEngine/`

### TCGA Databases

The TCGA Raw databases are required only to run the code of the PLOS article. Full data is downloaded from UCSC Xena Functional Genomics Browser.

Find here: https://xenabrowser.net/datapages/?hub=https://gdc.xenahubs.net:443.

Specifically, `star_counts.tsv`, `mirna.tsv` and `survival.tsv` are required fo each database.

All three files should be located inside the folder of the specific cancer in its directory: `Library/PLOS Code/TCGA Raw Data/BRCA` (for example). The file should be saved with **its original name**.

## Quickstart

Let start with a basic example. This example is coded in `Examples/QuickStart.ipynb`. See `PLOS Code` for the article code.

Firstly, we will import the required modules.

```python

import pandas as pd

from sklearn.model_selection import train_test_split

import miRNetClassifier

```

Load the dataset *SampleData/SampleDataBRCA* and prepare the data. More example datasets could be found in *SampleData* folder.

```python

# Load the dataset.

data_brca = pd.read_csv('SampleData/SampleDataBRCA.csv', sep = ',', index_col = 0)

# Select expression data (X) and the class (y)

X = data_brca.drop('classvalues', axis = 1)
y = data_brca['classvalues']

# Split the dataset in train and test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

```

The `MRNC` estimator has two parameters: `n_con` and `precision`. This two parameters will define the number of interactions of the model and the precision in the functional information estimation process.

The most simple usage of the package is reduced to fit the model to data with `fit()` and show the related network with `get_network()`. By default, the model will be fitted and the network represented with the default number of interactions.

```python

# Initialize estimator with default parameters (precision = 10, n_con = 20)

mrnc = miRNetClassifier.MRNC()

# Fit the model.

mrnc.fit(X_train, y_train)

# Display the coregulation network.

mrnc.get_network()

```
The resulting network:

<p align="center">
  <img src="network.png" alt="Alt Text" width="600" height="600">
</p>

Once the model is fitted, several atributes could be accessed.

```python

# Show the structural information.

print(mrnc.structural_information_)

# Show the functional information.

print(mrnc.functional_information_)

# The included micros in the CLG model.

print(mrnc.micros_)

# The included genes in the CLG model.

print(mrnc.genes_)

# The ordered list of interactions with their scores.

print(mrnc.connections_)

# The parameters of the model.

print(mrnc.clgc_)

# The graph associated to the CLG model.

print(mrnc.G_)

```

`mrnc.G_` loads the network as a MultDiGraph object of `networkx` package, allowing to use any of its methods. For example:

```python

# See nodes

print(mrnc.G_.nodes())

# See edges

print(mrnc.G_.edges())

# Add nodes.

mrnc.G_.add_node('hsa-mir-21')

mrnc.G_.add_node('hsa-mir-22')

# Add edges.

mrnc.G_.add_edge('hsa-mir-21', 'hsa-mir-22')

# Remove node and associated edges.

mrnc.G_.remove_node('hsa-mir-21')

```

A first sight of the network can be easily drawn using networkx methods.

```python

import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout

pos = graphviz_layout(mrnc.G_, prog='neato')

nx.draw_networkx_nodes(mrnc.G_, pos, node_color='skyblue', node_size=200)

nx.draw_networkx_edges(mrnc.G_, pos)

nx.draw_networkx_labels(mrnc.G_, pos, font_size=8, font_color='black')

```

<p align="center">
  <img src="output.png" alt="Alt Text" width="400" height="300">
</p>

This representation lacks important information, such as the functional and structural information between nodes. The `get_network()` framework introduced by `RNACOREX` offers a more complete visualization of the network.

The fitted model can also be used for predictions.

```python

# Predict

mrnc.predict(X_test)

# Predict proba

mrnc.predict_proba(X_test)

```

Instead of doing all the fitting process at one step, it can be done in more steps, allowing a bigger personalization of the model.

```python

# Initialize estimator with default parameters (precision = 10, n_con = 20)

mrnc = miRNetClassifier.MRNC()

# Initialize model and calculate structural information

mrnc.initialize_model(X_train, y_train)

# Calculate functional information. compute_functional() allows to use different X_train and y_train than the ones used for initializing the model.
# This permits, for example, fitting models with different train and test sets without having to initialize the model again.
# The used X_train and y_train sets should have the same variables (microRNA and mRNA) as the ones used in initialize_model().

mrnc.compute_functional()

# Fit the model

mrnc.fit_only()

# Get the network

mrnc.get_network()

```

The fit-predict framework executes the model from an static point of view, with a specific number of interaction defined by the user.

The `structure_search()` function allows to develop networks with 1 to *max_models* interactions and obtain its metrics. The model has to be initialized and the functional information computed beforehand. Train and test samples can be specified or not depending on user requirements. If train sets are not specified the ones included in `initialize_model()` are used. If train sets are specified the variables have to coincide with the ones used in `initialize_model()`.

```python

# Search for best structure with new train and test sets.

mrnc.structure_search(X_train, y_train, X_test, y_test, 100)

# Search for best structure with new train set and without test set.

mrnc.structure_search(X_train, y_train, max_models = 100)

# Search for best structure with previously initialized train set and test set.

mrnc.structure_search(X_test = X_test, y_test = y_test, max_models = 100)

# Search for best structure with only previously initialized train set.

mrnc.structure_search(max_models = 100)

# Show metrics of all networks

print(mrnc.structure_metrics_)

```

`structure_metrics_` displays metrics for all models from 1 to *max_models* interactions, showing *Accuracy*, *Log-Likelihood*, *BIC*, *AUC*, *Sensitivity* and *Specificity* of each network.

## References

[1] Lewis BP, Shih IH, Jones-Rhoades MW, Bartel DP. Prediction of mammalian MicroRNA targets. Cell 2003; 115, 787-798. \
[2] Maragkakis M, Reczko M, Simossis VA, Alexiou P, Papadopoulos GL, Dalamagas T, Giannopoulos G, Goumas G, Koukis E, Kourtis K, Vergoulis T, Koziris N, Sellis T, Tsanakas P, Hatzigeorgiou AG. DIANA-microT web server: elucidating microRNA functions through target prediction. Nucleic Acids Res. 2009 Jul;37(Web Server issue):W273-6. doi: 10.1093/nar/gkp292. Epub 2009 Apr 30. PMID: 19406924; PMCID: PMC2703977. \
[3] Chou CH, Shrestha S, et al. miRTarBase update 2018: a resource for experimentally validated microRNA-target interactions. Nucleic Acids Res. 2018 Jan 4;46(D1):D296-D302.
