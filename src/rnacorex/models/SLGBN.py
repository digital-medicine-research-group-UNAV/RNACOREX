import numpy as np
import networkx as nx
import numpy as np
from scipy.stats import norm
from pygam import LinearGAM, s, te
from functools import reduce
import operator
from joblib import Parallel, delayed
from itertools import combinations
import time


def learn_SLGBN(X, filtered_dag, classes, info, filtered_fmi, filtered_smi, G):
    """
    Learns the parameters of a Structural Learning Bayesian Network (SLGBN)
    for each class and evaluates its log-likelihood.

    Computes the network structure, estimates class-specific parameters,
    calculates class prior probabilities and individual likelihoods, and
    evaluates the total log-likelihood. Stores the learned parameters,
    network structure, metadata, and evaluation results in a dictionary.

    Args:
        X (pd.DataFrame): Gene expression dataset.
        filtered_dag (np.ndarray): Filtered directed acyclic graph adjacency matrix.
        classes (np.ndarray): One-hot encoded class labels for each individual.
        info (dict): Dictionary containing node metadata, including labels,
            types, and names.
        filtered_fmi (np.ndarray): Filtered Functional Mutual Information matrix.
        filtered_smi (np.ndarray): Filtered Structural Mutual Information matrix.
        G (nx.Graph): NetworkX graph representing the network structure.

    Returns:
        dict: Dictionary containing the learned class priors, model parameters,
            standard deviations, network structure, node metadata, mutual
            information matrices, log-likelihood, and updated graph.
    """
    SLGBN_structure = structure_SLGBN(X, filtered_dag, info)

    final_G = SLGBN_structure[0]
    node_types = SLGBN_structure[1]
    node_types_list = list(node_types.values())

    # Learn the network parameters for each class.
    params = parameters_SLGBN(X, classes, final_G, node_types)

    k = classes.shape[1]
    m = X.shape[0]

    sigma = np.array([
        [params[c][node]['sigma'] for c in range(k)]
        for node in params[0]
    ])

    classp = np.zeros(k)

    for c in range(0, k):
        classp[c] = np.sum(classes[:, c], axis=0)
        classp[c] /= m

    # Compute the class prior probabilities P(C=c) for each class c.

    # Compute P(X | class).
    pdf = np.zeros((m, k))

    for c in range(k):
        # Compute the probability of each individual given the parameters
        # of class c and the final network structure.
        pdf[:, c] = pdf_slgbn(
            X,
            params[c],
            final_G
        )

    # Multiply the likelihoods by the corresponding class prior probabilities.
    posterior_unnormalized = pdf * classp[np.newaxis, :]

    # Retain the probabilities corresponding to the true class of each individual.
    prob_ll = classes * posterior_unnormalized

    # Sum the probabilities across classes for each individual.
    # Since only the true class contributes, this retains its probability.
    sumprob = np.sum(prob_ll, axis=1)

    # Sum the log-probabilities of all individuals to obtain the total log-likelihood.
    ll = np.sum(np.log(np.clip(sumprob, 1e-300, None)))

    clgc = {
        "classp": classp,
        "params": params,
        "beta": None,
        "sigma": sigma,
        "covmat": None,
        "dag": filtered_dag,
        "labels": info["labels"],
        "types": info["types"],
        "distribution": node_types_list,
        "names": info["names"],
        "fmi": filtered_fmi,
        "smi": filtered_smi,
        "ll": ll,
        "acc": None,
        "bic": None
    }

    # Add the node types as attributes to the graph.
    label_to_name = dict(zip(info['labels'], info['names']))

    node_types_graph = {
        label_to_name[label]: node_type
        for label, node_type in node_types.items()
    }

    nx.set_node_attributes(G, node_types_graph, name='type')

    clgc['G'] = G

    return clgc



def estimate_parameters(train_data, G, node_types, n_bases, penalization, params_prev=None, nodes_to_update=None, score_cache=None):

    """
    Estimate local conditional distributions for each node given its parents.

    Supports:
    - Linear Gaussian models
    - Nonlinear models using GAM (Generalized Additive Models)

    Key ideas:
    - Each node is modeled conditionally on its parents.
    - Linear: least squares regression.
    - Nonlinear: spline-based GAM.
    - Uses caching to avoid recomputing parameters.

    Args:
        train_data (pd.DataFrame): Training dataset.
        G (nx.DiGraph): Current graph structure.
        node_types (dict): Mapping node -> "linear" or "nonlinear".
        n_bases (int): Number of spline bases (for nonlinear).
        penalization (float): Regularization parameter for GAM.
        params_prev (dict): Previous parameters (for incremental updates).
        nodes_to_update (list): Nodes to recompute.
        score_cache (dict): Cache for parameter reuse.

    Returns:
        dict: Estimated parameters per node.
    """

    if params_prev is None:
        params = {}
    else:
        params = params_prev.copy()

    if nodes_to_update is None:
        nodes_to_update = list(G.nodes())

    for node in nodes_to_update:

        # Initialize dictionary for the node
        if node not in params:
            params[node] = {}

        # Parents
        parents = sorted(G.predecessors(node))
        node_type = node_types[node]

        key = (node, tuple(parents), node_type)

        subset = train_data

        y = subset[node].values

        if score_cache is not None and key in score_cache:
            params[node] = score_cache[key]["params"]
            continue

        if len(parents) == 0:

            if node_type in ["linear", "nonlinear"]:

                beta = np.array([np.mean(y)])
                sigma2 = np.var(y)

                params[node] = {
                    "type": node_type,
                    "beta": beta,
                    "sigma": sigma2,
                    "model": None
                }

            continue

        else:
        
            X = subset[parents].values

            if node_type == "linear":
                X_lin = np.column_stack([np.ones(X.shape[0]), X])
                beta = np.linalg.lstsq(X_lin, y, rcond=None)[0]
                sigma2 = np.sum((y - X_lin @ beta)**2) / (X_lin.shape[0]) # MLE
                # sigma2 = np.sum((y - X_lin @ beta)**2) / (X_lin.shape[0] - X_lin.shape[1]) # Corrected

                params[node] = {
                    "type": "linear",
                    "beta": beta,
                    "sigma": sigma2,
                    "model": None
                }

            elif node_type == "nonlinear":

                if n_bases is None:

                    terms = [s(i, spline_order=3, lam=penalization) for i in range(len(parents))]

                else:

                    terms = [s(i, n_splines=n_bases, spline_order=3, lam=penalization) for i in range(len(parents))]

                # if len(parents) > 1:
                #     interaction_terms = [te(i, j, n_splines=4) for i, j in combinations(range(len(parents)), 2)]
                #     terms.extend(interaction_terms)
                    
                terms = reduce(operator.add, terms)
                gam = LinearGAM(terms).fit(X, y)
                mu = gam.predict(X)
                sigma2 = np.sum((y - mu)**2) / len(y)

                params[node] = {
                    "type": "nonlinear",
                    "beta": None,
                    "sigma": sigma2,
                    "model": gam
                }

        if score_cache is not None:
            score_cache[key] = {
                "params": params[node]
            }

    return params

def compute_total_ll_bic(train_data, params, G, method = 'BIC', k_cache = None, ll_cache = None, nodes = None, spar = 0.75):

    """
    Compute total log-likelihood and penalized scores for the model.

    Supports:
    - LL (log-likelihood)
    - BIC
    - EBIC

    Key ideas:
    - Computes node-wise likelihood and aggregates.
    - Uses caches for efficiency (incremental updates).
    - Penalizes model complexity based on number of parameters.

    Args:
        train_data (pd.DataFrame): Dataset.
        params (dict): Node parameters.
        G (nx.DiGraph): Graph structure.
        method (str): Scoring method ("LL", "BIC", "EBIC").
        k_cache (dict): Cached parameter counts.
        ll_cache (dict): Cached likelihoods.
        nodes (list): Subset of nodes to recompute.
        spar (float): EBIC sparsity parameter.

    Returns:
        tuple: (score, updated_ll_cache, updated_k_cache)
    """

    total_ll = 0.0
    total_params = 0
    n_samples = len(train_data)

    if ll_cache is None:
        ll_cache_temp = {}
    
    else:
        ll_cache_temp = ll_cache.copy()

    if k_cache is None:
        k_cache_temp = {}

    else:
        k_cache_temp = k_cache.copy()

    if nodes is None:

        nodes_to_update = params.keys()

    else:
        
        # print(f"Computing LL for nodes: {nodes}")
        nodes_to_update = set(nodes)

    for node, node_params in params.items():

        # if nodes is None:
            # print(f"Computing LL for node: {node}")

        if node not in nodes_to_update:
            continue

        # Parents (sorted for consistent ordering)
        parents = sorted(G.predecessors(node))

        p = node_params

        node_ll = 0.0

        if p["beta"] is not None:
            # Caso lineal o no-lineal sin padres
            y_obs = train_data[node].values
            beta = p["beta"]
            sigma2 = p["sigma"]

            if len(parents) > 0:
                X = train_data[parents].values
                X = np.column_stack([np.ones(X.shape[0]), X])  # add intercept
            else:
                X = np.ones((len(y_obs), 1))

            mu = X @ beta  # vectorized dot product
            ll = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((y_obs - mu)**2) / sigma2
            node_ll += ll.sum()
            k_node = len(parents) + 2

        else:           
            if p["type"] == "nonlinear":

                gam = p["model"]
                edf = gam.statistics_['edof']
                k_node = gam.statistics_['edof'] + 1
                sigma2 = p["sigma"]
                mu_test = gam.predict(train_data[parents].values)
                y_test = train_data[node].values
                ll_test = -0.5*np.log(2*np.pi*sigma2) - 0.5*((y_test - mu_test)**2)/sigma2
                node_ll = ll_test.sum()

        ll_cache_temp[node] = node_ll
        k_cache_temp[node] = k_node

        # if nodes is None:
            # print(f" Node {node}: LL = {node_ll}")
    
    total_ll = sum(ll_cache_temp.values())
    total_params = sum(k_cache_temp.values())

    if method == 'LL':
        
        return total_ll, ll_cache_temp, k_cache_temp
    
    elif method == 'EBIC':

        total_ll -= 0.5 * total_params * np.log(n_samples) + 2 * spar * len(G.edges) * np.log(len(G.nodes))

        return total_ll, ll_cache_temp, k_cache_temp

    elif method == 'BIC':

        total_ll -= 0.5 * total_params * np.log(n_samples)

        return total_ll, ll_cache_temp, k_cache_temp
    
    else:

        print('PLEASE SELECT A VALID SCORE')


def evaluate_event(event, G, node_types, params, n_bases, penalization, data_train, k_nodos, ll_nodos, score, score_cache):
    
    """
    Evaluate a candidate modification (event) to the current model.

    Supported events:
    - change_type: change node type (linear ↔ nonlinear)
    - add_edge: add directed edge
    - remove_edge: remove edge
    - reverse_edge: reverse edge direction

    Key ideas:
    - Applies the modification to a copy of the graph.
    - Re-estimates only affected nodes.
    - Computes updated score.
    - Returns all necessary information for selection.

    Returns:
        tuple: (score, event, updated_types, updated_params, k_cache, ll_cache, tabu_move, updated_graph)
    """

    tipo = event[0]
    
    if tipo == "change_type":
        G_copy = G.copy()
        node_types_copy = node_types.copy()
        node = event[1]
        new_type = event[2]
        node_types_copy[node] = new_type
        params_prov = estimate_parameters(data_train, G_copy, node_types_copy, n_bases, penalization, params, [node], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[node])
        tabu = ("change_type", node)
        return ll, event, node_types_copy, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "add_edge":
        node, target = event[1], event[2]
        if nx.has_path(G, target, node):
            return -1e10, None, node_types, params, k_nodos, ll_nodos, None, G
        G_copy = G.copy()
        G_copy.add_edge(node, target)
        # if not nx.is_directed_acyclic_graph(G_copy):
            # return -1e10, None, node_types, params, ll_nodos
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[target])
        tabu = ("remove_edge", node, target)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "remove_edge":
        G_copy = G.copy()
        node, target = event[1], event[2]
        G_copy.remove_edge(node, target)
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[target])
        tabu = ("add_edge", node, target)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "reverse_edge":
        G_copy = G.copy()
        node, target = event[1], event[2]
        G_copy.remove_edge(node, target)
        if nx.has_path(G_copy, node, target):
            return -1e10, None, node_types, params, k_nodos, ll_nodos, None, G
        G_copy.add_edge(target, node)
        # if not nx.is_directed_acyclic_graph(G_copy):
            # return -1e10, None, node_types, params, ll_nodos
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [node, target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[node, target])
        tabu = ("reverse_edge", target, node)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy


def can_change_type(node, tabu, type_change, n_parents):

    """
    Check whether a node type change is allowed.

    Conditions:
    - Not in tabu list.
    - Type change is enabled.
    - Nonlinear type requires at least one parent.
    """

    can_try_type_change = True

    if ("change_type", node) in tabu:

        can_try_type_change = False

    if type_change == False:
                    
        can_try_type_change = False

    if type_change == 'nonlinear' and n_parents == 0:

        can_try_type_change = False

    return can_try_type_change


def structure_SLGBN(X, dag, info):

    """
    Learns the structure and node types of a Structural Learning Bayesian
    Network (SLGBN) using an iterative optimization procedure.

    Initializes a directed graph from the input adjacency matrix and
    iteratively evaluates changes to node types, selecting the changes
    that maximize the chosen score. Uses tabu search and a patience
    criterion to control the optimization process.

    Args:
        X (pd.DataFrame): Gene expression dataset, with samples as rows
            and node identifiers as columns.
        dag (np.ndarray): Adjacency matrix representing the initial
            directed acyclic graph.
        info (dict): Dictionary containing node metadata, including
            the 'labels' key with the ordered node identifiers.

    Returns:
        tuple: A tuple containing:
            - nx.MultiDiGraph: The optimized network structure.
            - dict: The final node types.
            - dict: The estimated parameters for the final network.
    """

    G_0 = nx.MultiDiGraph()

    for i, node in enumerate(info["labels"]):
        G_0.add_node(node)

    for i in range(dag.shape[0]):  # Row names
        for j in range(dag.shape[1]):  # Column names
            if dag[i, j] == 1:  # If there's a 1 in the matrix
                G_0.add_edge(info["labels"][i], info["labels"][j])

    max_iter = 1000 
    iteration = 0
    tabu_size = 5
    patience = 5
    n_bases = 6
    count = 0
    penalization = 0.5
    score = 'EBIC'
    score_cache = None
    tabu_moves = []
    type_change = 'nonlinear'
    cores = 1

    nodes = X.columns
    node_types = {node: 'linear' for node in nodes}
    params = estimate_parameters(X, G_0, node_types, n_bases, penalization, score_cache)
    total_ll, ll_nodos, k_nodos = compute_total_ll_bic(X, params, G_0, method = score)

    best_G = G_0.copy()
    best_node_types = node_types.copy()
    best_params = params.copy()
    best_ll = total_ll

    while iteration < max_iter:

        best_event = None

        events_to_try = []

        for i, node in enumerate(nodes):

            n_parents = G_0.in_degree(node)

            if can_change_type(node, tabu_moves, type_change, n_parents):

                current_type = node_types[node]

                new_type = type_change if current_type == 'linear' else 'linear'

                events_to_try.append(("change_type", node, new_type))

        if not events_to_try:

            print("No events to try. Stopping algorithm.")
            
            final_G = best_G
            final_node_types = best_node_types
            final_params = best_params
    
            break

        results = Parallel(n_jobs=cores)( 
            delayed(evaluate_event)(event, G_0, node_types, params, n_bases, penalization, X, k_nodos, ll_nodos, score, score_cache)
            for event in events_to_try
        )

        best_result = max(results, key=lambda x: x[0])

        ll_select, best_event, node_types_select, params_select, k_nodos_select, ll_nodos_select, tabu, G_prov = best_result

        
        tabu_moves.append(tabu)

        if len(tabu_moves) > tabu_size:
            tabu_moves.pop(0)


        params = params_select
        G_0 = G_prov.copy()
        ll_nodos = ll_nodos_select
        k_nodos = k_nodos_select
        node_types = node_types_select

        iteration += 1

        if ll_select > best_ll:
            count = 0
            best_params = params
            best_node_types = node_types_select
            best_G = G_0.copy()
            best_ll = ll_select

        else:
            count += 1

        # if best_event[0] == 'change_type':
        #     print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}. LL = {ll_select}. Count = {count}")
        # else:
        #     print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}, {best_event[2]}: LL = {ll_select}. Count = {count}")

        if count >= patience:

            final_G = best_G
            final_node_types = best_node_types
            final_params = best_params

            break

    return final_G, final_node_types, final_params


def parameters_SLGBN(X, classes, G, node_types):

    """
    Estimates the parameters of a Structural Learning Bayesian Network
    (SLGBN) separately for each class.

    Selects the samples belonging to each class and estimates the
    network parameters using the given graph structure and node types.

    Args:
        X (pd.DataFrame or np.ndarray): Gene expression dataset, with
            samples as rows and nodes as columns.
        classes (np.ndarray): One-hot encoded class labels for each sample.
        G (nx.MultiDiGraph): Network structure.
        node_types (dict): Dictionary mapping each node to its type.

    Returns:
        dict: Dictionary mapping each class index to its estimated
            network parameters.
    """

    n_bases = 6
    penalization = 0.5

    n_classes = classes.shape[1]

    params = {}

    for c in range(n_classes):

        mask = classes[:, c] == 1

        X_class = X.loc[mask] if hasattr(X, "loc") else X[mask]

        params[c] = estimate_parameters(
            X_class,
            G,
            node_types,
            n_bases,
            penalization
        )

    return params



def pdf_slgbn(data, class_params, dag):

    """
    Computes the likelihood of each sample given the parameters and
    structure of a Structural Learning Bayesian Network (SLGBN).

    Converts the network structure to an adjacency matrix if necessary
    and computes the conditional probability density of each node given
    its parents. Supports linear and nonlinear models and combines the
    node-wise probability densities to obtain the likelihood of each sample.

    Args:
        data (pd.DataFrame): Gene expression dataset, with samples as rows
            and nodes as columns.
        class_params (dict): Dictionary containing the estimated parameters
            for each node, including regression coefficients or nonlinear
            models, variance parameters, and node types.
        dag (nx.DiGraph, nx.MultiDiGraph, or np.ndarray): Network structure,
            represented as a directed graph or an adjacency matrix.

    Returns:
        np.ndarray: Array containing the joint likelihood of each sample
            given the network structure and model parameters.
    """

    m = data.shape[0]

    labels = list(data.columns)

    # Si se recibe un grafo de NetworkX, convertirlo a matriz
    if isinstance(dag, (nx.DiGraph, nx.MultiDiGraph)):

        dag_matrix = nx.to_numpy_array(
            dag,
            nodelist=labels,
            dtype=int
        )

    else:

        dag_matrix = np.asarray(dag)

    n = dag_matrix.shape[1]

    pdf = np.ones(m)

    for node in range(n):

        node_label = labels[node]
        node_params = class_params.get(node_label, class_params.get(node))

        if node_params is None:
            raise KeyError(
                f"Parameters for node {node_label!r} were not found. "
                f"Available keys: {list(class_params.keys())[:5]}"
            )

        parents = np.where(dag_matrix[:, node] == 1)[0]
        parent_labels = sorted(labels[parent] for parent in parents)

        y_obs = data.iloc[:, node].values
        sigma2 = max(float(node_params["sigma"]), 1e-12)

        # --- LINEAL ---
        if node_params["beta"] is not None:

            beta = node_params["beta"]

            if len(parents) > 0:
                X = data[parent_labels].values
                X = np.column_stack([np.ones(m), X])
            else:
                X = np.ones((m, 1))

            mu = X @ beta

        # --- NO LINEAL ---
        elif node_params["type"] == "nonlinear":

            if len(parents) == 0:
                raise ValueError(f"Node {node_label} has no parents")

            mu = node_params["model"].predict(
                data[parent_labels].values
            )

        else:
            raise ValueError(
                f"Unsupported node type: {node_params['type']}"
            )

        pdf *= norm.pdf(
            y_obs,
            loc=mu,
            scale=np.sqrt(sigma2)
        )

    return pdf