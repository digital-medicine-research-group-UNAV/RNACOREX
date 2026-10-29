import numpy as np
from scipy.special import logsumexp
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from .models.SLGBN import learn_SLGBN
from .models.SPBN import learn_SPBN
from .models.LGBN import learn_LGBN
# from models.SLGBN import learn_SLGBN
# from models.SPBN import learn_SPBN
# from models.LGBN import learn_LGBN


def fit_model(X, y, smi, fmi, n_con, ranking, model, info):

    """
    Fits a probabilistic network model using a ranked set of interactions.

    Builds a directed acyclic graph (DAG) from the interaction ranking,
    filters the graph and associated data, and fits the specified model.

    Args:
        X (np.ndarray): Input feature matrix.
        y (np.ndarray): Target variable containing the class labels.
        smi (np.ndarray): Structure Mutual Information matrix.
        fmi (np.ndarray): Function Mutual Information matrix.
        n_con (int): Number of connections to include in the network.
        ranking (list): Ranked list of candidate interactions.
        model (str): Model to fit ('LGBN', 'SLGBN', or 'SPBN').
        info (dict): Dictionary containing node information, including
            names and types.

    Returns:
        dict: Fitted model containing the learned network parameters and
        associated metadata.
    """

    binary_matrix, coordinates = build_dag_from_ranking(
        ranking,
        smi.shape[0],
        smi.shape[1],
        n_con
    )

    filtered_dag, filtered_fmi, filtered_smi, info, X, classes = filter_dag(
        X,
        y,
        smi,
        fmi,
        binary_matrix,
        coordinates,
        info
    )

    filtered_dag, G = get_network(info, filtered_dag)

    if model == 'LGBN':

        clgc = learn_LGBN(
            X,
            classes,
            filtered_dag,
            filtered_fmi,
            filtered_smi,
            info
        )

        clgc['G'] = G
        clgc['model'] = 'LGBN'

    elif model == 'SLGBN':

        clgc = learn_SLGBN(
            X,
            filtered_dag,
            classes,
            info,
            filtered_fmi,
            filtered_smi,
            G
        )

        clgc['model'] = 'SLGBN'

    elif model == 'SPBN':

        clgc = learn_SPBN(
            X,
            filtered_dag,
            classes,
            info,
            filtered_fmi,
            filtered_smi,
            G
        )

        clgc['model'] = 'SPBN'

    return clgc


def build_ranking(smi, fmi, mode='alternating', weight=0.5, ties='isolated'):
    """
    Builds a complete ranking of interactions based on structural and
    functional mutual information scores.

    Identifies all interactions present in either the SMI or FMI matrix
    and ranks them according to the selected mode and tie-breaking
    criterion.

    Args:
        smi (np.ndarray): Structural Mutual Information matrix.
        fmi (np.ndarray): Functional Mutual Information matrix.
        mode (str, optional): Ranking strategy. Options are 'structural',
            'functional', 'weighted', and 'alternating'. Defaults to
            'alternating'.
        weight (float, optional): Weight assigned to SMI in the combined
            score for weighted ranking. The FMI weight is calculated as
            1 - weight. Defaults to 0.5.
        ties (str, optional): Tie-breaking criterion for structural and
            weighted ranking. Options are 'isolated', 'connected', and
            'functional'. Defaults to 'isolated'.

    Returns:
        list: Ranked interactions, where each element contains the row
        index, column index, SMI score, FMI score, and combined degree
        of the connected nodes, in the format
        [i, j, s, f, neighbors].
    """

    # Conectividad de los nodos
    row_counts = np.count_nonzero(smi, axis=1)
    col_counts = np.count_nonzero(smi, axis=0)
    degrees = row_counts + col_counts

    # Obtener únicamente las posiciones con interacción
    rows, cols = np.nonzero((smi != 0) | (fmi != 0))

    connects = []

    for i, j in zip(rows, cols):
        s = smi[i, j]
        f = fmi[i, j]
        vecinos = degrees[i] + degrees[j]

        connects.append([i, j, s, f, vecinos])

    # ---------- ORDEN SEGÚN MODO ----------

    if mode == 'structural':

        if ties == 'isolated':
            connects.sort(key=lambda x: (-x[2], x[4]))

        elif ties == 'connected':
            connects.sort(key=lambda x: (-x[2], -x[4]))

        elif ties == 'functional':
            connects.sort(key=lambda x: (-x[2], -x[3]))

        else:
            raise ValueError("Opción 'ties' no válida.")

    elif mode == 'functional':

        connects.sort(key=lambda x: -x[3])

    elif mode == 'weighted':

        combined = []

        for i, j, s, f, v in connects:
            combined_score = weight * s + (1 - weight) * f
            combined.append([i, j, s, f, combined_score, v])

        if ties == 'connected':
            combined.sort(key=lambda x: (-x[4], -x[5]))

        elif ties == 'isolated':
            combined.sort(key=lambda x: (-x[4], x[5]))

        elif ties == 'functional':
            combined.sort(key=lambda x: (-x[4], -x[3]))

        else:
            raise ValueError("Opción 'ties' no válida.")

        connects = [
            [i, j, s, f, v]
            for i, j, s, f, _, v in combined
        ]

    elif mode == 'alternating':

        if ties == 'isolated':
            connects_s = sorted(
                connects,
                key=lambda x: (-x[2], x[4])
            )

        elif ties == 'connected':
            connects_s = sorted(
                connects,
                key=lambda x: (-x[2], -x[4])
            )

        elif ties == 'functional':
            connects_s = sorted(
                connects,
                key=lambda x: (-x[2], -x[3])
            )

        else:
            raise ValueError("Opción 'ties' no válida.")

        connects_f = sorted(
            connects,
            key=lambda x: -x[3]
        )

        interleaved = [
            item
            for pair in zip(connects_s, connects_f)
            for item in pair
        ]

        seen = set()
        ranking = []

        for i, j, s, f, v in interleaved:
            key = (i, j)

            if key not in seen:
                ranking.append([i, j, s, f, v])
                seen.add(key)

        connects = ranking

    else:
        raise ValueError(
            "Modo no válido. Usa: "
            "'structural', 'functional', 'weighted' o 'alternating'."
        )

    return connects



def build_dag_from_ranking(ranking, n_rows, n_cols, n_con):
    """
    Builds a directed acyclic graph (DAG) representation from a ranked
    list of interactions.

    Selects the top-ranked interactions up to the specified limit and
    creates a binary adjacency matrix and a list of selected coordinates.

    Args:
        ranking (list): Ranked interactions, where each element contains
            the row index, column index, SMI score, FMI score, and
            combined degree of the connected nodes.
        n_rows (int): Number of rows in the adjacency matrix.
        n_cols (int): Number of columns in the adjacency matrix.
        n_con (int): Maximum number of top-ranked interactions to select.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Binary matrix representing the selected
              interactions.
            - list: Coordinates of the selected interactions, with each
              element represented as [i, j].
    """

    selected = ranking[:n_con]

    binary_matrix = np.zeros(
        (n_rows, n_cols),
        dtype=int
    )

    coordinates = []

    for i, j, _, _, _ in selected:
        binary_matrix[i, j] = 1
        coordinates.append([i, j])

    return binary_matrix, coordinates



def filter_dag(X, y, smi, fmi, dag, coordinates, info):

    """
    Filters the DAG, mutual information matrices, metadata, and expression
    data to retain only the nodes involved in the selected interactions.

    Extracts the unique node indices from the interaction coordinates,
    filters the corresponding matrices and metadata, selects the relevant
    features from the input data, and encodes the target labels using
    one-hot encoding.

    Args:
        X (pd.DataFrame): Input expression data, with feature names as
            index labels or column identifiers accessible through
            info['labels'].
        y (np.ndarray or list): Target class labels.
        smi (np.ndarray): Structural Mutual Information matrix.
        fmi (np.ndarray): Functional Mutual Information matrix.
        dag (np.ndarray): Adjacency matrix representing the DAG.
        coordinates (list): Coordinates of selected interactions, with
            each element represented as [i, j].
        info (dict): Dictionary containing node metadata, including
            'labels', 'types', and 'names'.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Filtered DAG adjacency matrix.
            - np.ndarray: Filtered FMI matrix.
            - np.ndarray: Filtered SMI matrix.
            - dict: Updated metadata containing only the selected nodes.
            - pd.DataFrame or np.ndarray: Filtered expression data.
            - np.ndarray: One-hot encoded target class labels.
    """

    # Filter DAG

    flattened = np.array([item for sublist in coordinates for item in sublist])
    flattened = np.unique(flattened).astype(int)
    filtered_dag = dag[np.ix_(flattened, flattened)]
    filtered_fmi = fmi[np.ix_(flattened, flattened)]
    filtered_smi = smi[np.ix_(flattened, flattened)]
   
    # Filter info

    info['labels'] = [info['labels'][i] for i in flattened]
    info['types'] = [info['types'][i] for i in flattened]
    info['names'] = [info['names'][i] for i in flattened]

    # Filter expression data

    X_filtered = X[info['labels']]

    # Construct classes object

    encoder = OneHotEncoder(sparse_output=False)
    classes = encoder.fit_transform(np.array(y).reshape(-1, 1))
    
    return filtered_dag, filtered_fmi, filtered_smi, info, X_filtered, classes



def get_network(info, filtered_dag):

    """
    Constructs a directed network from a filtered adjacency matrix and
    removes cycles using a greedy heuristic.

    Creates a directed multigraph with nodes colored according to their
    types, adds edges based on the adjacency matrix, and iteratively
    removes edges from detected cycles by selecting the edge whose target
    node has the highest total degree. The corresponding entries in the
    adjacency matrix are updated accordingly.

    Args:
        info (dict): Dictionary containing node metadata, including
            'names' and 'types'.
        filtered_dag (np.ndarray): Binary adjacency matrix representing
            the filtered directed graph.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Updated adjacency matrix after cycle removal.
            - nx.MultiDiGraph: Directed multigraph with typed node colors
              and no remaining directed cycles.
    """

    G = nx.MultiDiGraph()

    color_map = {

        "miRNA": "lightgreen", 
        "mRNA": "skyblue",
        "lncRNA": "lightcoral",
        "default": "gray"
    }

    for i, node in enumerate(info["names"]):
        node_type = info["types"][i] 
        color = color_map.get(node_type, color_map["default"]) 
        G.add_node(node, color=color)

    # Iterate over all rows and columns in the binary matrix
    for i in range(filtered_dag.shape[0]):  # Row names
        for j in range(filtered_dag.shape[1]):  # Column names
            if filtered_dag[i, j] == 1:  # If there's a 1 in the matrix
                G.add_edge(info["names"][i], info["names"][j])  # Add edge from row_node to col_node

    # CHECK FOR CYCLES

    # Te criterion for breaking cycles is the Greedy Heuristic Approach, eliminating 
    # first those edges in the cycle with the higher degree in the target node.

    # find_cycle() uses a Depth-First Search (DFS) backtracking approach for finding cycles.
    # Similar to Tarjan's Strongly Connected Components (SCC) algorithm but finding only a
    # single cycle.
    
    try:
        while True:
            cycle = list(nx.find_cycle(G, orientation="original"))  # Find a cycle
            if not cycle:
                break

            # Select the edge whose target node has the highest degree
            max_degree = -1
            edge_to_remove = None

            for edge in cycle:
                source, target = edge[:2]
                degree = G.in_degree(target) + G.out_degree(target)  # or just G.degree(target)
                if degree > max_degree:
                    max_degree = degree
                    edge_to_remove = (source, target)

            G.remove_edge(*edge_to_remove)  # Remove the selected edge

            # Update external matrix if needed
            edge_origin = info['names'].index(edge_to_remove[0])
            edge_target = info['names'].index(edge_to_remove[1])
            filtered_dag[edge_origin][edge_target] = 0

            print(f"Removed edge (target with highest degree): {edge_to_remove}")

    except nx.NetworkXNoCycle:
        pass

    return filtered_dag, G
