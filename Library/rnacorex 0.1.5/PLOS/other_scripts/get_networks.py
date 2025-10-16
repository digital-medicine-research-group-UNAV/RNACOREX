import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd


bbdds = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

for bd in bbdds:

    cons = pd.read_csv(f"../results/connections_lognorm_{bd}.csv")

    G = nx.MultiDiGraph()

    nodes = set(cons.iloc[:, 0]).union(set(cons.iloc[:, 1]))

    for node in nodes:
            
        if str(node).startswith('hsa'):

            color = "lightgreen"

        else:
                
            color = 'skyblue'

        G.add_node(node, color=color)

    for i in range(len(cons)):

        src = cons.iloc[i, 0]
        tgt = cons.iloc[i, 1]
        G.add_edge(src, tgt)

    plt.figure(figsize=(16, 16))
    nodes = G.nodes
    pos = graphviz_layout(G, prog='neato')
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    plt.axis('off')
    plt.savefig(f"Results/network_{bd}.eps", format="eps", dpi=300, bbox_inches='tight')
    plt.close()
