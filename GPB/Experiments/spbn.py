import sys
import importlib
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from rnacorex import RNACOREX


datasets = [
    "BRCA",
    "LUAD",
    "SKCM",
    "UCEC"
]

ks = range(2,121)

output_file = "spbn_results.csv"

if os.path.exists(output_file):
    os.remove(output_file)

first_write = True

accuracies = {}
aucs = {}

for cancer in datasets:

    print(f"\n{'='*50}")
    print(f"Procesando {cancer}")
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

    rnacorex_spbn = RNACOREX(
            n_con=(2, 120),
            precision=10,
            model="SPBN"
        )

    rnacorex_spbn.fit(X_train, y_train)

    with open(f"Models/{cancer}/spbn.pkl", "wb") as f:

        pickle.dump(rnacorex_spbn, f)

    rl_spbn = rnacorex_spbn.predict(X_test)

    clas_spbn = rl_spbn['classification']

    acc_spbn = (np.asarray(clas_spbn) == np.asarray(y_test)).mean(axis=1)
        
    post_spbn = rl_spbn['posterior']

    y_score_spbn = [arr[:, 1] for arr in post_spbn]

    auc_spbn = np.array([roc_auc_score(y_test, scores) for scores in np.asarray(y_score_spbn)])

    rows = []

    for i, acc in enumerate(acc_spbn):
        rows.append({
            'bd': cancer,
            'k': list(ks)[i],
            'accuracy': acc,
            'auc': auc_spbn[i]
        })

    df_cancer = pd.DataFrame(rows)

    df_cancer.to_csv(
        output_file,
        mode='w' if first_write else 'a',
        header=first_write,
        index=False
    )

    first_write = False

    print(f"{cancer} guardado correctamente.")