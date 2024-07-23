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
import CLGStructure
import FuncInformation
import StrucInformation

```

```python

# Load the dataset.

data_brca = pd.read_csv('SampleData/SampleDataBRCA.csv', sep = ',', index_col = 0)

# Select expression data (X) and the class (y)

X = data_brca.drop('classvalues', axis = 1)
y = data_brca['classvalues']

# Split the dataset in train and test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

```

miRNetClassifier uses the scikit-learn fit-predict schema. Before fitting, the model has to be initialized with `initialize_model()`, the functional information calculated with `initialize_model()` and the interaction ranking constructed with `interaction_ranking()`.

```python

# Initialize estimator with default parameters (precision = 10, n_con = 20)

mrnc = miRNetClassifier.MRNC()

# Initialize model and calculate structural information

mrnc.initialize_model(X_train, y_train)

# Calculate functional information

mrnc.compute_functional(X_train, y_train)

# Compute the interaction ranking

mrnc.interaction_ranking()

```

```python

# Fit model

mrnc.fit(X_train, y_train)

# Predict

mrnc.predict(X_test)

# Predict proba

mrnc.predict_proba(X_test)

```

```python

# Structure search

mrnc.structure_search(X_train, y_train, X_test, y_test, 100)

```

```python

# Coregulation network

mrnc.show_connections()

```

## References
