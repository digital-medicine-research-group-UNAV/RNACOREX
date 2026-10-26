import copy
import numpy as np
import networkx as nx
from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)



def learn_SPBN(X, filtered_dag, classes, info, filtered_fmi, filtered_smi, G):
    
    model = structure_SPBN(X, filtered_dag, info)

    params = parameters_SPBN(X, classes, model)

    k = classes.shape[1]
    m = X.shape[0]

    # sigma = np.array([[params[c][node]['sigma'] for c in range(k)] for node in params[0]])

    classp = np.zeros(k)

    for c in range(0,k):
        classp[c] = np.sum(classes[:,c], axis=0)
        classp[c] /= m

    # classp representa el prior de la clase, es decir, P(C=c) para cada clase c.

    logpdf = np.zeros((m, k))

    # Se calcula la probabilidad de cada individuo dado los parámetros de la clase c.

    for c in range(k):

        logpdf[:, c] = logpdf_spbn(
            X,
            params[c]
        )

    # --------------------------------------------------
    # log P(C=c) + log P(X | C=c)
    # --------------------------------------------------

    log_prior = np.log(
        np.clip(classp, 1e-300, None)
    )

    log_joint = (
        logpdf +
        log_prior[np.newaxis, :]
    )

    # --------------------------------------------------
    # Seleccionar la clase real de cada muestra
    # --------------------------------------------------

    # classes tiene un 1 en la clase real
    log_prob_real_class = classes * log_joint

    # Para cada muestra queda únicamente
    # log P(X_i, C_i)
    log_prob_real_class = np.sum(
        log_prob_real_class,
        axis=1
    )

    # --------------------------------------------------
    # Log-Likelihood total
    # --------------------------------------------------

    ll = np.sum(log_prob_real_class)

    clgc = {"classp":classp,"params": params,"beta":None,"sigma":None,"covmat":None,"dag":filtered_dag,"labels":info["labels"],"types":info["types"],"distribution":None,"names":info["names"],"fmi":filtered_fmi,"smi":filtered_smi,"ll":ll,"acc":None,"bic":None}

    node_types = {}

    for node in model.nodes():
        cpd = clgc['params'][0].cpd(node)
        node_types[node] = type(cpd).__name__

    node_distribution = [
        'linear' if value == 'LinearGaussianCPD' else 'nonlinear'
        for value in node_types.values()
    ]

    clgc['distribution'] = node_distribution

    nx.set_node_attributes(G, node_types, name='type')

    label_to_name = dict(zip(info['labels'], info['names']))

    node_types_graph = {
        label_to_name[label]: node_type
        for label, node_type in node_types.items()
    }

    nx.set_node_attributes(G, node_types_graph, name='type')

    clgc['G'] = G

    return clgc



def structure_SPBN(X, filtered_dag, info):

    from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)

    # Hay que darle de entrada la estructura inicial.

    hc = GreedyHillClimbing()

    pool = OperatorPool([ChangeNodeTypeSet()])

    vl = ValidatedLikelihood(X, k=10, seed=0)

    node_types = [(name, LinearGaussianCPDType()) for name in X.columns]

    arcs = [
        (X.columns[i], X.columns[j])
        for i in range(filtered_dag.shape[0])
        for j in range(filtered_dag.shape[1])
        if filtered_dag[i, j] == 1
    ]

    start_model = SemiparametricBN(list(X.columns), arcs, node_types)

    model = hc.estimate(pool, vl, start_model, patience=5)

    return model


def parameters_SPBN(X, classes, model):

    from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)

    # Hay que darle de entrada la estructura inicial.

    priors = {}

    models = {}

    n_classes = classes.shape[1]

    for c in range(n_classes):

        mask = classes[:, c] == 1
        X_c = X[mask]
        priors[c] = len(X_c) / len(X)

        model_c = copy.deepcopy(model)

        model_c.fit(X_c)

        models[c] = model_c

    # Models incluye un modelo por clase. La estructura es común y los parámetros se aprenden por clase. 

    return models



def logpdf_spbn(data, class_params):

    scores = []

    for i in range(len(data)):

        x = data.iloc[i].to_frame().T

        loglik = class_params.slogl(x)

        score = np.asarray(loglik, dtype=float).ravel()[0]

        scores.append(score)

    return np.asarray(scores, dtype=float)