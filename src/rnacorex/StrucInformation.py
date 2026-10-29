import numpy as np
import pandas as pd
from tqdm import tqdm


def consistency_index(filtered_engine, labels):

    """
    Calculates consistency indices between pairs of interaction engines
    based on the overlap of their predicted gene interactions.

    For each pair of engines, computes the number of interactions in the
    larger set, the number of shared interactions, and a consistency
    index adjusted for the total number of possible gene pairs. Negative
    values are set to zero, and the resulting matrix is normalized by
    the sum of its elements.

    Args:
        filtered_engine (pd.DataFrame): Interaction database containing
            the columns 'Engine', 'Gene1', and 'Gene2'.
        labels (list): List of gene identifiers used to determine the
            total number of possible gene pairs.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Normalized consistency index matrix comparing
              pairs of engines.
            - list: Names of the interaction engines in the order used
              in the matrix.
    """

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

    """
    Constructs adjacency matrices for each interaction engine based on
    the interactions in the filtered database.

    For each engine, creates a binary adjacency matrix using the provided
    gene labels as node indices and marks the corresponding entries for
    each interaction. Each matrix is stored alongside its engine name.

    Args:
        filtered_engine (pd.DataFrame): Filtered interaction database
            containing the columns 'Engine', 'Gene1', and 'Gene2'.
        engines (list): List of interaction engine names to process.
        labels (list): List of gene identifiers defining the node order
            in the adjacency matrices.

    Returns:
        list: A list of pairs, each containing a binary adjacency matrix
        (np.ndarray) and the corresponding engine name.
    """
    
    smi = np.zeros((len(labels), len(labels)))
    contribution_mask = np.zeros_like(smi, dtype=bool) 

    presence_mask = np.zeros_like(smi, dtype=bool)

    for dag, _ in dags:
        presence_mask |= (dag == 1)

    for i in range(len(engines)):
        dag1 = dags[i][0]
        for j in range(len(engines)):
            dag2 = dags[j][0]

            if not np.isnan(ci[i][j]) and ci[i][j] > 0:
                indices = np.where((dag1 == 1) & (dag2 == 1))
                smi[indices] += ci[i][j]
                contribution_mask[indices] = True

    smi[(presence_mask) & (~contribution_mask)] = 0.0001

    return smi



def SMI(X, info, engine):

    """
    Computes the Structural Mutual Information (SMI) matrix from multiple
    interaction engines and their consistency indices.

    Extracts gene labels from the metadata, calculates the consistency
    indices between interaction engines, constructs an adjacency matrix
    for each engine, and uses these matrices and consistency indices to
    compute the structural information matrix.

    Args:
        X (pd.DataFrame): Gene expression dataset, with samples as rows
            and gene identifiers as columns.
        info (dict): Dictionary containing gene metadata, including the
            'labels' key with the ordered gene identifiers.
        engine (pd.DataFrame): Interaction database containing the
            interactions and engine identifiers.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The input expression dataset.
            - np.ndarray: Structural Mutual Information matrix.
    """

    labels = info.get('labels')

    ci, engines = consistency_index(engine, labels)

    dags = engine_dags(engine, engines, labels)

    smi = structural_information(dags, engines, ci, labels)

    return X, smi