# miRNetClassifier

miRNetClassifier is a package that develops Conditional Linear Gaussian Classifiers (CLGs) to identify miRNA-mRNA coregulation networks and classify new samples.

## Requirements

## Quickstart

Let start with a basic example. This example is coded in *Examples/quickstart_1.py*. See *Examples* folder for additional examples.

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

As a first step, the model has to be initialized with `initialize_model()`, the functional information calculated with `compute_functional()` and the interaction ranking constructed with `interaction_ranking()`. By default, the `compute_functional()` uses the 

```python

# Initialize estimator with default parameters (precision = 10, n_con = 20)

mrnc = miRNetClassifier.MRNC()

# Initialize model and calculate structural information

mrnc.initialize_model(X_train, y_train)

# Calculate functional information

mrnc.compute_functional()

# Compute the interaction ranking

mrnc.interaction_ranking()

```

After executing this three functions, several atributes could be accessed.

```python

# Show the structural information.

print(mrnc.structural_information_)

# Show the functional information.

print(mrnc.functional_information_)

# The considered micros for the CLG model.

print(mrnc.micros_)

# The considered genes for the CLG model.

print(mrnc.genes_)

# The ordered list of interactions with their scores.

print(mrnc.connections_)

```

Once the model is initialized it could be fitted and used for predictions. The parameters of the model could be obtained through the `clgc_` atribute.

The `fit()` function does not need `X_train`, `y_train` as these are computed by `initialize_model()`. It does make not sense implementing `fit()` with other `X_train`, `y_train` sets other than the ones calculated with `initialize_model()`.

```python

# Fit the model

mrnc.fit()

# Predict

mrnc.predict(X_test)

# Predict proba

mrnc.predict_proba(X_test)

# Obtain the parameters of the model

print(mrnc.clgc_)

```

With `show_connections()` the coregulation network is displayed. The displayed network will have the number of interactions defined in the `n_con` atribute of the estimator.

```python

# Coregulation network.

mrnc.show_connections()

```

The fit-predict framework executes the model from an static point of view, with a specific number of interaction defined by the udes.

The `structure_search()` function allows to develop networks with 1 to *max_models* interactions and obtain its metrics. A test sample can be specified or not depending on user requirements.

```python

mrnc.structure_search(X_train, y_train, X_test, y_test, 100)

mrnc.structure_search(X_train, y_train, max_models = 100)

```

## References
