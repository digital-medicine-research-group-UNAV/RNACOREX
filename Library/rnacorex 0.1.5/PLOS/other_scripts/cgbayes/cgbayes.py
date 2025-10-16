import rnacorex
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

# Code for CGBayesNets benchmarking using RNACOREX. Currently dor AUC metric.

databases = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

# summary_df = pd.DataFrame(columns=['bd', 'best_k', 'acc_resust', 'acc_test']) # For accuracy metric.
summary_df = pd.DataFrame(columns=['bd', 'best_k', 'auc_resust', 'auc_test'])

for bd in databases:

    data_train = pd.read_csv(f'../../data/cgbayesnets/data_train_{bd}_lognorm.csv', sep = ',', index_col = 0)
    data_test = pd.read_csv(f'../../data/cgbayesnets/data_test_{bd}_lognorm.csv', sep = ',', index_col = 0)

    X_train = data_train.drop('Class', axis = 1)
    y_train = data_train['Class']

    X_test = data_test.drop('Class', axis = 1)
    y_test = data_test['Class']

    mrnc = rnacorex.MRNC()

    mrnc.initialize_model(X_train, y_train)
    mrnc.compute_functional()
    mrnc.rank()

    results = []

    for k in tqdm(range(2, 200)):
        mrnc.n_con = k
        mrnc.fit_only()
        preds = mrnc.predict(X_train)
        accuracy_train = np.mean(y_train.values == preds)
        probs_train = mrnc.predict_proba(X_train)[:,1]
        auc_train = roc_auc_score(y_train, probs_train)
        # results.append((k, accuracy_train)) # For accuracy metric.
        results.append((k, auc_train))

    # best_k, best_accuracy = max(results, key=lambda x: x[1]) # For accuracy metric.
    best_k, best_auc = max(results, key=lambda x: x[1])

    mrnc.n_con = best_k
    mrnc.fit_only()
    preds = mrnc.predict(X_test)
    accuracy_test = np.mean(y_test.values == preds)
    probs_test = mrnc.predict_proba(X_test)[:,1]
    auc_test = roc_auc_score(y_test, probs_test)

    summary_df = pd.concat([
        summary_df,
        # pd.DataFrame([{'bd': bd, 'best_k': best_k, 'acc_resust': best_accuracy, 'acc_test': accuracy_test}]) # For accuracy metric.
        pd.DataFrame([{'bd': bd, 'best_k': best_k, 'auc_resust': best_auc, 'auc_test': auc_test}])
    ], ignore_index=True)

    print(f"Database: {bd}, Best k: {best_k}, Train AUC: {best_auc:.4f}, Test AUC: {auc_test:.4f}")

    summary_df.to_csv(f'../results/cgbayes_auc_{bd}.csv', index=False)
    # summary_df.to_csv('../results/cgbayes_acc_{bd}.csv', index=False) # For accuracy metric.
