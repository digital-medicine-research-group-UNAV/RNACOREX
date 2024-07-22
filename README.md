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

#from CRFE._crfe import CRFE   
sys.path.insert(0, "../") 
from CRFE._crfe import CRFE 

```
