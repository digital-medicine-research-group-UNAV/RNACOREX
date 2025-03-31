import numpy as np
import pandas as pd


def consistency_index(filtered_engine, labels):

    # Calculate consistency indices for each pair of engines.

    engines = list(filtered_engine['Engine'].unique())

    n = len(labels) * len(labels)

    k = np.zeros((len(engines), len(engines)))
    r = np.zeros((len(engines), len(engines)))
    ci = np.zeros((len(engines), len(engines)))

    for i in range(0, len(engines)):

            gi = engines[i]

            for j in range((i+1), len(engines)):

                gj = engines[j]

                if gi != gj:
                    
                    ki = filtered_engine['Engine'].value_counts().get(gi, 0)
                    kj = filtered_engine['Engine'].value_counts().get(gj, 0)
                    kij = np.maximum(ki, kj)

                    k[i][j] = kij

                    ri = set(zip(filtered_engine[filtered_engine['Engine'] == gi]['Gene1'], filtered_engine[filtered_engine['Engine'] == gi]['Gene2']))
                    rj = set(zip(filtered_engine[filtered_engine['Engine'] == gj]['Gene1'], filtered_engine[filtered_engine['Engine'] == gj]['Gene2']))

                    shared_interactions = ri.intersection(rj)
                    rij = len(shared_interactions)

                    r[i][j] = rij

                    ci[i][j] = (rij * n - kij**2) / (kij * (n - kij))

                    if ci[i][j] < 0: ci[i][j] = 0

    ci = ci / np.sum(ci)

    return ci, engines



def engine_dags(filtered_engine, engines, labels):
     
    dags = []

    for engine in engines:

        engine_data = filtered_engine[filtered_engine['Engine'] == engine]

        dag_engine = np.zeros((len(labels), len(labels)))

        for _, interaction in engine_data.iterrows():
            
            element1 = interaction['Gene1']
            element2 = interaction['Gene2']

            index1 = labels.index(element1)
            index2 = labels.index(element2)

            dag_engine[index1][index2] = 1
        
        dags.append([dag_engine, engine])

    return dags



def structural_information(dags, engines, ci, labels):
     
    smi = np.zeros((len(labels), len(labels)))

    for i in range(len(engines)):

        dag1 = dags[i][0]

        smi += dag1 * 0.0001
        
        for j in range(len(engines)):
            
            dag2 = dags[j][0]

            if not np.isnan(ci[i][j]):

                indices = np.where((dag1 == 1) & (dag2 == 1))

                smi[indices] += ci[i][j]

    return smi



def SMI(X, info, engine):

    labels = info.get('labels')

    ci, engines = consistency_index(engine, labels)

    dags = engine_dags(engine, engines, labels)

    smi = structural_information(dags, engines, ci, labels)

    return X, smi