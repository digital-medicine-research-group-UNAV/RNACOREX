import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Get the path to the miRNetClassifier directory
miRNetClassifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'miRNetClassifier'))

# Add the miRNetClassifier path to sys.path
sys.path.append(miRNetClassifier_path)

# Define a function for display and save results in graphics.

def plot_results_cv(accuracy_results, notebook, rank = False):

    fig, axs = plt.subplots(2, 2, figsize=(16,12), gridspec_kw={'hspace': 0.3})

    x = np.arange(len(accuracy_results))
    axs[0, 0].plot(x, accuracy_results['Test'], label='Test', c = 'blue', linewidth=2)
    axs[0, 0].fill_between(x, accuracy_results['Test'] - accuracy_results['STD Test'], accuracy_results['Test'] + accuracy_results['STD Test'], alpha=0.15, color='blue')
    axs[0, 0].plot(x, accuracy_results['Random Forest'], color='green', label='Random Forest')
    axs[0, 0].set_title('vs Random Forest')
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].plot(x, accuracy_results['Test'], label='Test', c = 'blue', linewidth=2)
    axs[0, 1].fill_between(x, accuracy_results['Test'] - accuracy_results['STD Test'], accuracy_results['Test'] + accuracy_results['STD Test'], alpha=0.15, color='blue')
    axs[0, 1].plot(x, accuracy_results['Gradient Boosting'], color='red', label='Gradient Boosting')
    axs[0, 1].set_title('vs Gradient Boosting')
    axs[0, 1].legend(fontsize=8)
    axs[1, 0].plot(x, accuracy_results['Test'], label='Test', c = 'blue', linewidth=2)
    axs[1, 0].fill_between(x, accuracy_results['Test'] - accuracy_results['STD Test'], accuracy_results['Test'] + accuracy_results['STD Test'], alpha=0.15, color='blue')
    axs[1, 0].plot(x, accuracy_results['SVM'], color='grey', label='SVM')
    axs[1, 0].set_title('vs SVM')
    axs[1, 0].legend(fontsize=8)

    accuracy_results['X'] = accuracy_results['X'].astype(str)
    accuracy_results = accuracy_results.sort_values(by=['Test_from_mean'], ascending=True)

    if rank == True:

        def remove_first_word(string):
            words = string.split()  # Split the string into a list of words
            return ' '.join(words[1:]) 

        rank = pd.DataFrame({
            'Accuracy Test': accuracy_results['Test'],
            'Accuracy RF': accuracy_results['Random Forest'],
            'Accuracy SVC': accuracy_results['SVM'],
            'Accuracy GB': accuracy_results['Gradient Boosting']
            })

        acc_ranking = rank.rank(axis=1, method='min', ascending=False)

        axs[1, 1].plot([1, 4], [0, 0], color='lightgray', linewidth=5)

        markers = ['o', 's', '^', 'D']
        for column, rank in acc_ranking.mean().items():
            axs[1, 1].plot(rank, 0, 'o', label=f'{remove_first_word(column)}', markersize=10)

        for rank in acc_ranking.mean():
            axs[1, 1].text(rank, 0.01, f'{rank:.2f}', ha='center', va='bottom', color='black', fontsize=10)

        axs[1, 1].set_xlim(1, 4)
        axs[1, 1].set_title('Mean Rank')
        axs[1, 1].set_xlabel('Rank')

        textstr = '\n'.join([f'{remove_first_word(column)}: {rank:.2f}' for column, rank in acc_ranking.mean().items()])
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)

        axs[1, 1].text(0.95, 0.85, textstr, transform=axs[1, 1].transAxes, fontsize=12,
                       verticalalignment='center', horizontalalignment='right', bbox=props)
        axs[1, 1].legend(loc='upper left', ncol=2, fontsize=12)
        plt.savefig('PLOS_Results/'+notebook+'_rank.svg')
    
    else:

        colors = ['red' if value < 0 else 'green' for value in accuracy_results['Test_from_mean']]
        axs[1, 1].scatter(x, accuracy_results['Test_from_mean'], c=colors, s=10)
        axs[1, 1].set_title('Model Accuracy')
        axs[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.savefig('PLOS_Results/'+notebook+'_plot.svg')


# Develop the main code.

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

    accuracy_results.to_csv('PLOS_Results/results_'+notebook+'.csv', index=False)

    plot_results_cv(accuracy_results, notebook, rank = True)
    plot_results_cv(accuracy_results, notebook, rank = False)

    mrnc.compute_functional(mrnc.X_, mrnc.y_)
    mrnc.fit_only()
    # mrnc.connections_.to_csv('PLOS_Results/connections_'+notebook+'.csv', index=False)
    mrnc.get_network(k=accuracy_results['Test'].idxmax(), save = 'PLOS_Results/network_'+notebook+'.csv')
    




