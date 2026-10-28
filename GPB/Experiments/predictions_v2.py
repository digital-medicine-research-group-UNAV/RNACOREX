from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
import numpy as np
import os

datasets = [
    "BRCA",
    "LUAD",
    "SKCM",
    "UCEC"
]

ks = range(2, 120)

results_lgbn = []
results_slgbn = []

for bd in datasets:

    data_test = pd.read_csv(
        f"../Data/test_{bd.lower()}_lognorm.csv",
        index_col=0
    )

    X_test = data_test.drop("Class", axis=1)
    y_test = data_test["Class"].to_numpy()

    # LGBN-V2
    with open(f"Models/{bd}/lgbn.pkl", "rb") as f:
        lgbn = pickle.load(f)

    rl_lgbn = lgbn.predict(X_test)

    clas_lgbn = np.asarray(rl_lgbn["classification"])
    post_lgbn = rl_lgbn["posterior"]

    acc_lgbn = (
        clas_lgbn == y_test
    ).mean(axis=1)

    auc_lgbn = np.array([
        roc_auc_score(y_test, posterior[:, 1])
        for posterior in post_lgbn
    ])

    for k, accuracy, auc in zip(ks, acc_lgbn, auc_lgbn):
        results_lgbn.append({
            "bd": bd,
            "k": k,
            "accuracy": accuracy,
            "auc": auc
        })

    # SLGBN-V2
    with open(f"Models/{bd}/slgbn.pkl", "rb") as f:
        slgbn = pickle.load(f)

    rl_slgbn = slgbn.predict(X_test)

    clas_slgbn = np.asarray(rl_slgbn["classification"])
    post_slgbn = rl_slgbn["posterior"]

    acc_slgbn = (
        clas_slgbn == y_test
    ).mean(axis=1)

    auc_slgbn = np.array([
        roc_auc_score(y_test, posterior[:, 1])
        for posterior in post_slgbn
    ])

    for k, accuracy, auc in zip(ks, acc_slgbn, auc_slgbn):
        results_slgbn.append({
            "bd": bd,
            "k": k,
            "accuracy": accuracy,
            "auc": auc
        })

os.makedirs("Metrics", exist_ok=True)

# Guardar un CSV por modelo
pd.DataFrame(results_lgbn).to_csv(
    "Metrics/lgbn_v2_results.csv",
    index=False
)

pd.DataFrame(results_slgbn).to_csv(
    "Metrics/slgbn_v2_results.csv",
    index=False
)

print("Results saved:")
print("- Metrics/lgbn_v2_results.csv")
print("- Metrics/slgbn_v2_results.csv")