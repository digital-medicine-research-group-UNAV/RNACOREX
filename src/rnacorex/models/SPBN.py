import copy
import numpy as np
import networkx as nx
from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)



def learn_SPBN(X, filtered_dag, classes, info, filtered_fmi, filtered_smi, G):
    """
    Learns a Structural Probabilistic Bayesian Network (SPBN) and evaluates
    its total log-likelihood.

    Constructs the network structure, estimates class-specific parameters,
    computes class prior probabilities and sample log-likelihoods, and
    identifies the conditional probability distribution associated with
    each node. Stores the model parameters, network metadata, evaluation
    results, and updated graph in a dictionary.

    Args:
        X (pd.DataFrame): Gene expression dataset, with samples as rows
            and node identifiers as columns.
        filtered_dag (np.ndarray): Filtered adjacency matrix representing
            the network structure.
        classes (np.ndarray): One-hot encoded class labels for each sample.
        info (dict): Dictionary containing node metadata, including
            labels, types, and names.
        filtered_fmi (np.ndarray): Filtered Functional Mutual Information matrix.
        filtered_smi (np.ndarray): Filtered Structural Mutual Information matrix.
        G (nx.Graph): NetworkX graph representing the network structure.

    Returns:
        dict: Dictionary containing class prior probabilities, estimated
            parameters, network structure, node metadata, distribution
            types, mutual information matrices, total log-likelihood,
            and the updated graph.
    """
    model = structure_SPBN(X, filtered_dag, info)

    params = parameters_SPBN(X, classes, model)

    k = classes.shape[1]
    m = X.shape[0]

    classp = np.zeros(k)

    for c in range(0, k):
        classp[c] = np.sum(classes[:, c], axis=0)
        classp[c] /= m

    logpdf = np.zeros((m, k))

    for c in range(k):
        logpdf[:, c] = logpdf_spbn(
            X,
            params[c]
        )

    log_prior = np.log(
        np.clip(classp, 1e-300, None)
    )

    log_joint = (
        logpdf +
        log_prior[np.newaxis, :]
    )

    log_prob_real_class = classes * log_joint

    log_prob_real_class = np.sum(
        log_prob_real_class,
        axis=1
    )

    ll = np.sum(log_prob_real_class)

    clgc = {
        "classp": classp,
        "params": params,
        "beta": None,
        "sigma": None,
        "covmat": None,
        "dag": filtered_dag,
        "labels": info["labels"],
        "types": info["types"],
        "distribution": None,
        "names": info["names"],
        "fmi": filtered_fmi,
        "smi": filtered_smi,
        "ll": ll,
        "acc": None,
        "bic": None
    }

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

    """
    Learns the structure of a Semiparametric Bayesian Network (SPBN)
    using a greedy hill-climbing algorithm.

    Initializes the network with the provided adjacency matrix, assigns
    a linear Gaussian conditional probability distribution to each node,
    and optimizes the network by evaluating node type changes using
    a validated likelihood score.

    Args:
        X (pd.DataFrame): Gene expression dataset, with samples as rows
            and node identifiers as columns.
        filtered_dag (np.ndarray): Adjacency matrix defining the initial
            network structure.
        info (dict): Dictionary containing node metadata.

    Returns:
        SemiparametricBN: The learned Bayesian network model.
    """

    from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)

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

    """
    Estimates the parameters of a Semiparametric Bayesian Network (SPBN)
    separately for each class.

    Selects the samples belonging to each class, creates a copy of the
    network model, and fits the model to the corresponding class-specific
    data. The network structure is shared across classes, while the
    parameters are estimated independently for each class.

    Args:
        X (pd.DataFrame): Gene expression dataset, with samples as rows
            and node identifiers as columns.
        classes (np.ndarray): One-hot encoded class labels for each sample.
        model (SemiparametricBN): Initialized Bayesian network structure
            to be fitted separately for each class.

    Returns:
        dict: Dictionary mapping each class index to its fitted
            Semiparametric Bayesian Network model.
    """

    from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)

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

    return models



def logpdf_spbn(data, class_params):

    """
    Computes the log-likelihood of each sample given a fitted
    Semiparametric Bayesian Network (SPBN).

    Evaluates the log-likelihood of each individual sample using the
    fitted model and returns the resulting scores as a NumPy array.

    Args:
        data (pd.DataFrame): Dataset containing the samples to evaluate,
            with samples as rows and nodes as columns.
        class_params (SemiparametricBN): Fitted Bayesian network model
            containing the parameters for a specific class.

    Returns:
        np.ndarray: One-dimensional array containing the log-likelihood
            of each sample under the specified model.
    """

    scores = []

    for i in range(len(data)):

        x = data.iloc[i].to_frame().T

        loglik = class_params.slogl(x)

        score = np.asarray(loglik, dtype=float).ravel()[0]

        scores.append(score)

    return np.asarray(scores, dtype=float)