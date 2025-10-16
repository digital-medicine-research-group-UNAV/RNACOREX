import rnacorex
import pandas as pd

bbdds = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

for bd in bbdds:

    data = pd.read_csv('../data/main_experiments/data_plos_' + bd + '_lognorm.csv', sep=',', index_col=0)
    X = data.drop(columns=['Class'])
    y = data['Class']

    results = pd.read_excel('../results/table_results_' + bd + '.xlsx', sheet_name='rnacorex')
    k = int(results['mean'].tail(1).values[0])

    mrnc = rnacorex.MRNC(n_con = k, precision = 20, ties='isolated')

    mrnc.fit(X, y)

    cons = mrnc.connections_.head(k)

    cons.to_csv('../results/connections_lognorm_' + bd + '.csv', index=False)