import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_auc_score

datasets = [
    "BRCA",
    "LUAD",
    "SKCM",
    "UCEC"
]

ks = range(2, 120)

results = []

for bd in datasets:

    data_test = pd.read_csv(
        f"../Data/test_{bd.lower()}_lognorm.csv",
        index_col=0
    )

    X_test = data_test.drop("Class", axis=1)
    y_test = data_test["Class"]

    for k in tqdm(ks, desc=f"Predicting {bd}"):

        with open(f"Models/{bd}/v1_{k}.pkl", "rb") as f:
            lgbn_v1 = pickle.load(f)

        y_pred = np.asarray(
            lgbn_v1.predict(X_test)
        ).ravel()

        accuracy = (y_pred == y_test.to_numpy()).mean()

        y_score = lgbn_v1.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_score)

        results.append({
            "bd": bd,
            "k": k,
            "accuracy": accuracy,
            "auc": auc
        })

os.makedirs("Metrics", exist_ok=True)

# Guardar todos los resultados
results_df = pd.DataFrame(results)

results_df.to_csv(
    "Metrics/lgbn_v1_results.csv",
    index=False
)

print(results_df.head())
print("Results saved in Metrics/lgbn_v1_results.csv")