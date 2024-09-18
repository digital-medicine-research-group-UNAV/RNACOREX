import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold

# Get the path to the miRNetClassifier directory
miRNetClassifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'miRNetClassifier'))

# Add the miRNetClassifier path to sys.path
sys.path.append(miRNetClassifier_path)

import miRNetClassifier

# notebook_names = ['brca', 'luad', 'hnsc', 'lgg', 'kirc', 'laml', 'lihc', 'coad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']
notebook_names = ['brca', 'luad', 'hnsc', 'lgg']

for notebook in tqdm(notebook_names):

    print(notebook.upper())

    data = pd.read_csv('Clean Data/SampleData'+notebook.upper()+'.csv', sep = ',', index_col = 0)    

    X = data.drop('classvalues', axis = 1)
    y = data['classvalues']

    mrnc = miRNetClassifier.MRNC()

    accuracies_train = []
    accuracies_test = []
    accuracies_test_rf = []
    accuracies_test_svc = []
    accuracies_test_knn = []
    accuracies_test_gb = []
    auc_test = []
    sensi_test = []
    speci_test = []

    mrnc.initialize_model(X, y)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)

    for fold, (train_indices, test_indices) in enumerate(skf.split(mrnc.X_, mrnc.y_)):

        print('Fold:', fold)

        X_train, y_train = mrnc.X_.iloc[train_indices, :], mrnc.y_.iloc[train_indices]
        X_test, y_test = mrnc.X_.iloc[test_indices, :], mrnc.y_.iloc[test_indices]

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        # Calculate functional information

        mrnc.compute_functional(X_train, y_train)

        # Compute the interaction ranking

        # mrnc.interaction_ranking()

        # Perform the structure search

        mrnc.structure_search(X_train, y_train, X_test, y_test, 100)

        accuracies_train.append(mrnc.structure_metrics_['Accuracy Train'])
        accuracies_test.append(mrnc.structure_metrics_['Accuracy Test'])
        accuracies_test_rf.append(mrnc.structure_metrics_['Accuracy RF'])
        accuracies_test_svc.append(mrnc.structure_metrics_['Accuracy SVC'])
        accuracies_test_knn.append(mrnc.structure_metrics_['Accuracy KNN'])
        accuracies_test_gb.append(mrnc.structure_metrics_['Accuracy GB'])
        auc_test.append(mrnc.structure_metrics_['AUC'])
        sensi_test.append(mrnc.structure_metrics_['Sensitivity'])
        speci_test.append(mrnc.structure_metrics_['Specificity'])

        # Show the dual connections

        # mrnc.show_dual_connections()
    
    data_array_test = np.array(accuracies_test)
    data_array_train = np.array(accuracies_train)
    data_array_rf = np.array(accuracies_test_rf)
    data_array_svc = np.array(accuracies_test_svc)
    data_array_knn = np.array(accuracies_test_knn)
    data_array_gb = np.array(accuracies_test_gb)
    data_array_auc = np.array(auc_test)
    data_array_sensi = np.array(sensi_test)
    data_array_speci = np.array(speci_test)

    average_test = np.mean(data_array_test, axis=0)
    average_rf = np.mean(data_array_rf, axis = 0)
    average_svc = np.mean(data_array_svc, axis = 0)
    average_knn = np.mean(data_array_knn, axis = 0)
    average_gb = np.mean(data_array_gb, axis = 0)
    average_auc = np.mean(data_array_auc, axis = 0)
    average_sensi = np.mean(data_array_sensi, axis = 0)
    average_speci = np.mean(data_array_speci, axis = 0)

    std_deviation_test = np.std(data_array_test, axis=0)
    std_deviation_rf = np.std(data_array_rf, axis=0)
    std_deviation_svc = np.std(data_array_svc, axis=0)
    std_deviation_knn = np.std(data_array_knn, axis=0)
    std_deviation_gb = np.std(data_array_gb, axis=0)

    accuracy_results = pd.DataFrame()
    accuracy_results['Test'] = average_test
    accuracy_results['STD Test'] = std_deviation_test
    accuracy_results['Random Forest'] = average_rf
    accuracy_results['SVM'] = average_svc
    accuracy_results['Gradient Boosting'] = average_gb

    accuracy_results['Mean_alternatives'] = accuracy_results[['Random Forest', 'SVM', 'Gradient Boosting']].mean(axis=1)
    accuracy_results['Max_alternatives'] = accuracy_results[['Random Forest', 'SVM', 'Gradient Boosting']].max(axis=1)
    accuracy_results['Min_alternatives'] = accuracy_results[['Random Forest', 'SVM', 'Gradient Boosting']].min(axis=1)
    accuracy_results['Test_from_mean'] = accuracy_results['Test'] - accuracy_results['Mean_alternatives']
    accuracy_results['Max_alternatives'] = accuracy_results['Max_alternatives'] - accuracy_results['Mean_alternatives']
    accuracy_results['Min_alternatives'] = accuracy_results['Min_alternatives'] - accuracy_results['Mean_alternatives']
    accuracy_results['Mean_alternatives'] = accuracy_results['Mean_alternatives'] - accuracy_results['Mean_alternatives']
    accuracy_results['X'] = range(100)
    accuracy_results['AUC'] = average_auc
    accuracy_results['Sensitivity'] = average_sensi
    accuracy_results['Specificity'] = average_speci

    # accuracy_results.to_csv('PLOS_Results/results_'+notebook+'.csv', index=False)

    # CLGStructure2.plot_results_cv(accuracy_results, notebook, rank = True)
    # CLGStructure2.plot_results_cv(accuracy_results, notebook, rank = False)

    mrnc.compute_functional(mrnc.X_, mrnc.y_)
    mrnc.fit_only()
    # mrnc.connections_.to_csv('PLOS_Results/connections_'+notebook+'.csv', index=False)
    mrnc.get_network(k=accuracy_results['Test'].idxmax())
    




