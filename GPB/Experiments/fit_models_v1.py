import pandas as pd
import pickle
import rnacorex
import os

# FOR THE EXECUTION OF THIS SCRIPT RNACOREX 0.1.5 IS REQUIRED

# PLEASE DO: python -m pip install RNACOREX==0.1.5

datasets = [
    "BRCA",
    "LUAD",
    "SKCM",
    "UCEC"
]

for cancer in datasets:

    print(f"\n{'='*50}")
    print(f"Processing {cancer}")
    print(f"{'='*50}")

    # Cargar datos
    data_train = pd.read_csv(
        f"../Data/train_{cancer.lower()}_lognorm.csv",
        index_col=0
    )

    data_test = pd.read_csv(
        f"../Data/test_{cancer.lower()}_lognorm.csv",
        index_col=0
    )

    X_train = data_train.drop("Class", axis=1)
    y_train = data_train["Class"]
    X_test = data_test.drop("Class", axis=1)
    y_test = data_test["Class"]

    # Crear carpeta
    output_dir = f"Models/{cancer}"
    os.makedirs(output_dir, exist_ok=True)

    mrnc = rnacorex.MRNC(precision=10)

    mrnc.initialize_model(X_train, y_train)
    mrnc.compute_functional()
    mrnc.rank()

    for k in range(2, 120):

        mrnc.n_con = k
        mrnc.fit_only()

        with open(f"{output_dir}/v1_{k}.pkl", "wb") as f:
            pickle.dump(mrnc, f)