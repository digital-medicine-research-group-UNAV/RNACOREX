import pandas as pd
import pickle
from rnacorex import RNACOREX
import os


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

    # -------------------------
    # SLGBN
    # -------------------------
    rnacorex_slgbn = RNACOREX(
        n_con=(2, 120),
        precision=10,
        model="SLGBN"
    )

    rnacorex_slgbn.fit(X_train, y_train)

    with open(f"{output_dir}/slgbn.pkl", "wb") as f:
        pickle.dump(rnacorex_slgbn, f)

    # -------------------------
    # LGBN
    # -------------------------
    rnacorex_lgbn = RNACOREX(
        n_con=(2, 120),
        precision=10
    )

    rnacorex_lgbn.fit(X_train, y_train)

    with open(f"{output_dir}/lgbn.pkl", "wb") as f:
        pickle.dump(rnacorex_lgbn, f)

    print(f"{cancer} terminado.")