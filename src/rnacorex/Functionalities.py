import networkx as nx
import numpy as np
import pandas as pd


def get_interactions(smi, fmi, mode, weight, ties, info, n):

    """
    Extracts and ranks network interactions based on Structural Mutual
    Information (SMI) and Functional Mutual Information (FMI).

    Identifies interactions with nonzero SMI or FMI values, associates
    each interaction with its node names and types, and sorts the
    interactions according to the selected ranking mode and tie-breaking
    criterion.

    Args:
        smi (np.ndarray): Structural Mutual Information matrix.
        fmi (np.ndarray): Functional Mutual Information matrix.
        mode (str): Ranking strategy. Options are 'structural',
            'functional', 'weighted', and 'alternating'.
        weight (float): Weight assigned to SMI in the combined score
            for weighted ranking. The FMI weight is 1 - weight.
        ties (str): Tie-breaking criterion. Options are 'isolated',
            'connected', and 'functional'.
        info (dict): Dictionary containing node metadata, including
            'names' and 'types'.
        n (int): Maximum number of top-ranked interactions to return.

    Returns:
        pd.DataFrame: DataFrame containing the top-ranked interactions
        with columns 'Node1', 'Node2', 'Type1', 'Type2', 'SMI', and
        'FMI'.
    """

    n_nodes = smi.shape[0]
    # Calcular grado de nodos (para desempates)
    row_counts = np.count_nonzero(smi, axis=1)
    col_counts = np.count_nonzero(smi, axis=0)
    degrees = row_counts + col_counts

    # Construir lista de interacciones con nombres y tipos
    interactions = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            s = smi[i, j]
            f = fmi[i, j]
            if s != 0 or f != 0:
                vecinos = degrees[i] + degrees[j]
                interactions.append({
                    "Node1": info["names"][i],
                    "Node2": info["names"][j],
                    "Type1": info["types"][i],
                    "Type2": info["types"][j],
                    "SMI": s,
                    "FMI": f,
                    "DegreeSum": vecinos
                })

    # ---------- ORDEN SEGÚN MODO ----------
    if mode == 'structural':
        if ties == 'isolated':
            interactions.sort(key=lambda x: (-x["SMI"], x["DegreeSum"]))
        elif ties == 'connected':
            interactions.sort(key=lambda x: (-x["SMI"], -x["DegreeSum"]))
        elif ties == 'functional':
            interactions.sort(key=lambda x: (-x["SMI"], -x["FMI"]))
        else:
            raise ValueError("Opción 'ties' no válida.")

    elif mode == 'functional':
        interactions.sort(key=lambda x: -x["FMI"])

    elif mode == 'weighted':
        for inter in interactions:
            inter["WeightedScore"] = weight * inter["SMI"] + (1 - weight) * inter["FMI"]
        if ties == 'connected':
            interactions.sort(key=lambda x: (-x["WeightedScore"], -x["DegreeSum"]))
        elif ties == 'isolated':
            interactions.sort(key=lambda x: (-x["WeightedScore"], x["DegreeSum"]))
        else:
            interactions.sort(key=lambda x: (-x["WeightedScore"], -x["FMI"]))

    elif mode == 'alternating':
        # Orden por SMI
        if ties == 'isolated':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], x["DegreeSum"]))
        elif ties == 'connected':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], -x["DegreeSum"]))
        elif ties == 'functional':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], -x["FMI"]))
        else:
            raise ValueError("Opción 'ties' no válida.")
        # Orden por FMI
        fmi_sorted = sorted(interactions, key=lambda x: -x["FMI"])

        # Alternar listas eliminando duplicados
        combined_sorted = []
        seen = set()
        for smi_item, fmi_item in zip(smi_sorted, fmi_sorted):
            for item in (smi_item, fmi_item):
                key = (item["Node1"], item["Node2"])
                if key not in seen:
                    combined_sorted.append(item)
                    seen.add(key)
        interactions = combined_sorted
    else:
        raise ValueError("Modo no válido. Usa: 'structural', 'functional', 'weighted' o 'alternating'.")

    for inter in interactions:
        if "DegreeSum" in inter:
            del inter["DegreeSum"]
        if "WeightedScore" in inter:
            del inter["WeightedScore"]

    # Retornar solo las primeras n interacciones
    return pd.DataFrame(interactions[:n])



def display(G, save_format = None):

    """
    Displays a network graph using a Graphviz-based layout and optionally
    saves the visualization to a file.

    Draws the graph nodes using their assigned colors, displays edges and
    node labels, and hides the plot axes. If a save format is specified,
    the figure is saved to a file named 'network' with the corresponding
    extension.

    Args:
        G (nx.Graph or nx.DiGraph or nx.MultiDiGraph): NetworkX graph to
            visualize. Each node must have a 'color' attribute.
        save_format (str or None, optional): File format in which to save
            the figure, such as 'png', 'pdf', or 'svg'. If None, the
            figure is displayed without being saved. Defaults to None.

    Returns:
        None
    """
    
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    # Crear figura solo si no se proporciona un eje

    pos = graphviz_layout(G, prog='neato')

    node_colors = [
        G.nodes[node]["color"]
        for node in G.nodes
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_color='black',
        ax=ax
    )

    ax.set_axis_off()

    if save_format is not None:

            filename = f"network.{save_format}"
            fig.savefig(
                filename,
                format=save_format,
                bbox_inches='tight'
            )
            print(f"Figure saved as: {filename}")

    plt.show()


def get_interactions(smi, fmi, mode, weight, ties, info, n):

    """
    Extracts and ranks network interactions based on Structural Mutual
    Information (SMI) and Functional Mutual Information (FMI).

    Identifies interactions with nonzero SMI or FMI values, associates
    each interaction with its node names and types, and sorts the
    interactions according to the selected ranking mode and tie-breaking
    criterion.

    Args:
        smi (np.ndarray): Structural Mutual Information matrix.
        fmi (np.ndarray): Functional Mutual Information matrix.
        mode (str): Ranking strategy. Options are 'structural',
            'functional', 'weighted', and 'alternating'.
        weight (float): Weight assigned to SMI in the combined score
            for weighted ranking. The FMI weight is 1 - weight.
        ties (str): Tie-breaking criterion for structural and weighted
            ranking. Options are 'isolated', 'connected', and
            'functional'.
        info (dict): Dictionary containing node metadata, including
            'names' and 'types'.
        n (int): Maximum number of top-ranked interactions to return.

    Returns:
        pd.DataFrame: DataFrame containing the top-ranked interactions
        with columns 'Node1', 'Node2', 'Type1', 'Type2', 'SMI', and
        'FMI'.
    """
    
    n_nodes = smi.shape[0]
    row_counts = np.count_nonzero(smi, axis=1)
    col_counts = np.count_nonzero(smi, axis=0)
    degrees = row_counts + col_counts

    interactions = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            s = smi[i, j]
            f = fmi[i, j]
            if s != 0 or f != 0:
                vecinos = degrees[i] + degrees[j]
                interactions.append({
                    "Node1": info["names"][i],
                    "Node2": info["names"][j],
                    "Type1": info["types"][i],
                    "Type2": info["types"][j],
                    "SMI": s,
                    "FMI": f,
                    "DegreeSum": vecinos
                })

    if mode == 'structural':
        if ties == 'isolated':
            interactions.sort(key=lambda x: (-x["SMI"], x["DegreeSum"]))
        elif ties == 'connected':
            interactions.sort(key=lambda x: (-x["SMI"], -x["DegreeSum"]))
        elif ties == 'functional':
            interactions.sort(key=lambda x: (-x["SMI"], -x["FMI"]))
        else:
            raise ValueError("Opción 'ties' no válida.")

    elif mode == 'functional':
        interactions.sort(key=lambda x: -x["FMI"])

    elif mode == 'weighted':
        for inter in interactions:
            inter["WeightedScore"] = weight * inter["SMI"] + (1 - weight) * inter["FMI"]
        if ties == 'connected':
            interactions.sort(key=lambda x: (-x["WeightedScore"], -x["DegreeSum"]))
        elif ties == 'isolated':
            interactions.sort(key=lambda x: (-x["WeightedScore"], x["DegreeSum"]))
        else:
            interactions.sort(key=lambda x: (-x["WeightedScore"], -x["FMI"]))

    elif mode == 'alternating':
        if ties == 'isolated':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], x["DegreeSum"]))
        elif ties == 'connected':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], -x["DegreeSum"]))
        elif ties == 'functional':
            smi_sorted = sorted(interactions, key=lambda x: (-x["SMI"], -x["FMI"]))
        else:
            raise ValueError("Opción 'ties' no válida.")
        fmi_sorted = sorted(interactions, key=lambda x: -x["FMI"])

        combined_sorted = []
        seen = set()
        for smi_item, fmi_item in zip(smi_sorted, fmi_sorted):
            for item in (smi_item, fmi_item):
                key = (item["Node1"], item["Node2"])
                if key not in seen:
                    combined_sorted.append(item)
                    seen.add(key)
        interactions = combined_sorted
    else:
        raise ValueError("Modo no válido. Usa: 'structural', 'functional', 'weighted' o 'alternating'.")

    for inter in interactions:
        if "DegreeSum" in inter:
            del inter["DegreeSum"]
        if "WeightedScore" in inter:
            del inter["WeightedScore"]

    return pd.DataFrame(interactions[:n])