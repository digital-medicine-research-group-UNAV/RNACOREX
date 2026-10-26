import networkx as nx
import numpy as np
import pandas as pd


def get_interactions(smi, fmi, mode, weight, ties, info, n):
    """
    Extrae las interacciones de la red, ordenándolas según SMI y FMI
    de acuerdo al modo y criterio de desempate.

    Args:
        smi (np.ndarray): Matriz de Structure Mutual Information.
        fmi (np.ndarray): Matriz de Function Mutual Information.
        info (dict): Diccionario con claves "names" y "types" de los nodos.
        n (int): Número de interacciones a retornar.
        mode (str, opcional): 'structural', 'functional', 'weighted', 'alternating' (por defecto)
        weight (float, opcional): Peso de SMI en modo 'weighted' (0–1)
        ties (str, opcional): 'isolated', 'connected', 'functional' (por defecto: 'isolated')

    Returns:
        pd.DataFrame: DataFrame con las interacciones ordenadas.
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
    Extrae las interacciones de la red, ordenándolas según SMI y FMI
    de acuerdo al modo y criterio de desempate.

    Args:
        smi (np.ndarray): Matriz de Structure Mutual Information.
        fmi (np.ndarray): Matriz de Function Mutual Information.
        info (dict): Diccionario con claves "names" y "types" de los nodos.
        n (int): Número de interacciones a retornar.
        mode (str, opcional): 'structural', 'functional', 'weighted', 'alternating' (por defecto)
        weight (float, opcional): Peso de SMI en modo 'weighted' (0–1)
        ties (str, opcional): 'isolated', 'connected', 'functional' (por defecto: 'isolated')

    Returns:
        pd.DataFrame: DataFrame con las interacciones ordenadas.
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