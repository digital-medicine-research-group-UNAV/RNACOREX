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

    # Construir DAG a partir del ranking ya calculado

    binary_matrix, coordinates = build_dag_from_ranking(
        ranking,
        smi.shape[0],
        smi.shape[1],
        n_con
    )

    # Filtrar DAG, datos, etc.
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
    Construye el ranking completo de interacciones según SMI/FMI.

    Returns
    -------
    ranking : list
        Lista de interacciones ordenadas según el criterio seleccionado.
        Cada elemento tiene:
        [i, j, s, f, vecinos]
    """

    n_rows, n_cols = smi.shape

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
    Builds DAG with a previously calculated ranking.
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

    # Filter DAG

    # flattened = [item for sublist in coordinates for item in sublist]
    # flattened = np.unique(flattened)
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
