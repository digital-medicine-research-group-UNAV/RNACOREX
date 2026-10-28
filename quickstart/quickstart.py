import pandas as pd
import rnacorex
from rnacorex import RNACOREX

data_train = pd.read_csv(f"../GPB/Data/train_brca_lognorm.csv", index_col=0)
data_test = pd.read_csv(f"../GPB/Data/test_brca_lognorm.csv", index_col=0)

X_train = data_train.drop("Class", axis=1)
y_train = data_train["Class"]
X_test = data_test.drop("Class", axis=1)
y_test = data_test["Class"]

# Train a model with 40 interactions.

rnacorex = RNACOREX(
        n_con=40
    )

rnacorex.fit(X_train, y_train)

# PREDICTION WITH COMPLETE EVIDENCE

rl = rnacorex.predict(X_test)

# GET FITTED MODEL

model = rnacorex.models_[40]

# GET NETWORK

G = rnacorex.models_[40]['G']
