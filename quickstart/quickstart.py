import pandas as pd
import rnacorex
from rnacorex import RNACOREX

print("Paquete importado desde:", rnacorex.__file__)
print("Clase importada desde:", RNACOREX.__module__)

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

# Train models from 20 to 40 interactions.

rnacorex_multi = RNACOREX(
        n_con=(20, 40)
    )

rnacorex_multi.fit(X_train, y_train)

# Train models with 20 and 40 interactions.

rnacorex_multi2 = RNACOREX(
        n_con=[20, 40]
    )

rnacorex_multi2.fit(X_train, y_train)

# PREDICTION WITH COMPLETE EVIDENCE

rl = rnacorex.predict(X_test)

# PREDICTION WITH PARTIAL EVIDENCE

X_test_partial = X_test.sample(frac=0.9, axis=1, random_state=42)

rl_part = rnacorex.predict(X_test_partial)

rnacorex.get_interactions(20)

# GET FITTED MODEL

model = rnacorex.models_[40]

# WHEN MORE THAN ONE MODEL HAS BEEN FITTED, YOU CAN ACCESS THE CORRESPONDING DATA BY SELECTING THE NUMBER OF INTERACTIONS

model25 = rnacorex_multi.models_[25]

model35 = rnacorex_multi.models_[35]

# RAW NETWORK

G = rnacorex.models_[40]['G']

rnacorex.display(G)

# MPC NETWORK

maps = rnacorex.mpc(
    display=True,             
    c=0,                
    k=40,
    th=0.25
)
