from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from rnacorex import RNACOREX

datasets = [
    "BRCA",
    "LUAD",
    "SKCM",
    "UCEC"
]

ks = range(2, 120)

spbn = pd.read_csv('spbn_results.csv')

accuracies = {}
aucs = {}

for bd in datasets:

    accuracies[bd] = {}
    aucs[bd] = {}

    data_test = pd.read_csv(f'../Data/test_{bd.lower()}_lognorm.csv', index_col=0)

    X_test = data_test.drop("Class", axis=1)
    y_test = data_test["Class"]

    with open(f"Models/{bd}/lgbn.pkl", "rb") as f:

        # LGBN

        lgbn = pickle.load(f)

        rl_lgbn = lgbn.predict(X_test)

        clas_lgbn = rl_lgbn['classification']

        acc_lgbn = (np.asarray(clas_lgbn) == np.asarray(y_test)).mean(axis=1)
        
        post_lgbn = rl_lgbn['posterior']

        y_score_lgbn = [arr[:, 1] for arr in post_lgbn]

        auc_lgbn = np.array([roc_auc_score(y_test, scores) for scores in np.asarray(y_score_lgbn)])

        accuracies[bd]['LGBN-V2'] = acc_lgbn

        aucs[bd]['LGBN-V2'] = auc_lgbn


    with open(f"Models/{bd}/slgbn.pkl", "rb") as f:

        # SLGBN

        slgbn = pickle.load(f)

        rl_slgbn = slgbn.predict(X_test)

        clas_slgbn = rl_slgbn['classification']

        acc_slgbn = (np.asarray(clas_slgbn) == np.asarray(y_test)).mean(axis=1)
        
        post_slgbn = rl_slgbn['posterior']

        y_score_slgbn = [arr[:, 1] for arr in post_slgbn]

        auc_slgbn = np.array([roc_auc_score(y_test, scores) for scores in np.asarray(y_score_slgbn)])

        accuracies[bd]['SLGBN-V2'] = acc_slgbn
        
        aucs[bd]['SLGBN-V2'] = auc_slgbn

    accuracies[bd]['LGBN-V1'] = []

    aucs[bd]['LGBN-V1'] = []

    print('PREDICTING V1')

    for k in tqdm(ks):

        with open(f"Models/{bd}/v1_{k}.pkl", "rb") as f:

            lgbn_v1 = pickle.load(f)

        rl_v1 = lgbn_v1.predict(X_test)

        clas = rl_v1

        acc_lgbn_v1 = (np.asarray(clas).ravel() == y_test).mean()

        post_lgbn_v1 = lgbn_v1.predict_proba(X_test)

        auc_lgbn_v1 = roc_auc_score(y_test, post_lgbn_v1[:,1])

        accuracies[bd]['LGBN-V1'].append(acc_lgbn_v1)
        
        aucs[bd]['LGBN-V1'].append(auc_lgbn_v1)


        results_spbn = spbn[spbn['bd'] == bd]