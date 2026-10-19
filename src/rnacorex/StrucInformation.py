import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm



def intersection_adjmatrix(g1, g2):

    """

    Adds two numbers together.

    Args:

            g1 (nx.DiGraph): The first graph.
            g2 (nx.DiGraph): The second graph.

            g1 and g2 have to be composed of the same nodes.

    Returns:

            intersec_matrix: (np.ndarray): Dichotomical matrix (True/False) of dimension (n+m)*(n+m) in which an element is True if the edge is common between graphs.

    """

    # Calculate the adjacency matrix of the first graph.

    adjmatrix1 = nx.adjacency_matrix(g1).todense()
    adjmatrix1 = adjmatrix1 > 0
    adjmatrix1 = np.maximum(adjmatrix1, adjmatrix1.T)

    # Calculate the adjacency matrix of the second graph.

    adjmatrix2 = nx.adjacency_matrix(g2).todense()
    adjmatrix2 = adjmatrix2 > 0
    adjmatrix2 = np.maximum(adjmatrix2, adjmatrix2.T)

    # Calculate the intersection of matrices.

    intersec_matrix = adjmatrix1 & adjmatrix2

    # Convert matrix to dichotmical (True/False).

    intersec_matrix = intersec_matrix > 0

    return intersec_matrix


def consistency_index(g1, g2, n, scale=0):

    """
    Calculates the consistency index between two graphs.

    - R. Armañanzas. Revealing post-transcriptional microRNA-mRNA regulations in Alzheimer's disease through ensemble graphs. 
    BMC Genomics. 2018 Sep 24;19(Suppl 7):668. doi: 10.1186/s12864-018-5025-y. PMID: 30255799; PMCID: PMC6157163.

    Args:

            g1 (nx.DiGraph): The first graph.
            g2 (nx.DiGraph): The second graph.
            n (int): Number of maximum possible connections. If there are only micro-gen connections: number of micros * number of genes.
            scale (int): Type of scaling to be applied.
                0: No scaling
                1: Scales the index from [-km,1] to [0,1]
                2: Scales the index from [-1,1] to [0,1]

    Returns:

            ci (float): Value of the consistency index.

    """

    # Number of edges in each graph.

    k1 = g1.number_of_edges()
    k2 = g2.number_of_edges()

    # Compute the intersection matrix between the two graphs.

    intersecadjmatrix = intersection_adjmatrix(g1,g2)

    # Common edges between the two graphs.

    r = (intersecadjmatrix.sum())/2

    # Maximum number of edges between the two graphs.

    km = np.maximum(k1,k2)

    # Consistency index.

    ci_top = r*n-km*km
    nkm = (n-km)*1.0
    ci_down = km*nkm
    ci = ci_top / ci_down

    # Scale the index from [-km,1] to [0,1].

    if scale == 1:
        ci = (ci+km)/(1+km)

    # Scale the index from [-1,1] to [0,1].

    if scale == 2:
        if ci < -1:
            # ci = -1
            ci = 0
        else:
            km = 1
            ci = (ci+km) / (1+km)

    return ci



def ensemble_graph(glist, c, consistency_scale = 0):

    """
    
    Calculates the structural mutual information matrix from a set of graphs that are passed to the function.
    The graphs are weighted using the result obtained in their consistency index.
    It is essential that the graphs passed to the function are formed by the same nodes and therefore HAVE THE SAME DIMENSION.
    
    Args:

        glist (list): List of n DAGs (nx.Digraph) with the same set of nodes.
        c (int): Maximum number of possible connections. If there are only micro-gen connections: number of micros * number of genes.
        consistency_scale (int): Type of scaling to be applied in the consistency index:
                0: No scaling
                1: Scales the index from [-km,1] to [0,1]
                2: Scales the index from [-1,1] to [0,1]
        
    Returns:

        structural_information (np.ndarray): Numpy matrix of dimension n_nodes x n_nodes containing the structural mutual information of each pair of nodes.
    
    """

    # Number of graphs.

    n_graphs = len(glist)

    # Initialization of the graph weighting matrix.

    wmatrix = np.zeros((n_graphs, n_graphs))

    # Calculating the consistency index for each pair of graphs.

    for i in range(0, n_graphs-1):

        gi = glist[i]

        for j in range((i+1), n_graphs):

            gj = glist[j]

            # Calculate the index.
                
            w = consistency_index(gi, gj, c, scale=consistency_scale)

            # Assign the index to the weighting matrix.

            wmatrix[i][j] = w

    # Scale the consistency indexes so that all add 1 and each element is in range [0,1].

    wmatrix = wmatrix/np.sum(wmatrix)

    wmatrix[np.isnan(wmatrix)] = 0

    # Generate the matrix for functional information.
    
    n_nodes = glist[0].number_of_nodes()
    edgewam = np.zeros((n_nodes, n_nodes))

    # Select each pair of graphs.

    for i in range(0, n_graphs-1):

        gi = glist[i]

        for j in range((i+1), n_graphs):

            gj = glist[j]

            # Calculate the intersection between graphs (common edges with minimum weight).

            intersectionadjmatrix = intersection_adjmatrix(gi,gj)

            # To the final matrix of connection weights, the value obtained above is added multiplied by the consistency index.

            edgewam = edgewam + intersectionadjmatrix * wmatrix[i][j]
    

    structural_information = np.triu(edgewam)

    return structural_information



def initialize_graph(inputFile, mirNames = None, geneNames = None):

    """
    
    Builds a directed graph with the structural relationships of a .csv file.
    You can indicate those genes and microRNAs that are considered relevant for the study through two .csv-s (mirNames and geneNames).
    If you do not indicate genes and microRNAs by default it initializes the graph with all the connections in the file.
    If microRNAs and genes are indicated, the function ignores all nodes that are not included in these files.

    Args:

        inputFile (pd.DataFrame): .csv file with the information of the micro-gen relationships. Each line must be a relationship, having the gene in the first column and the microRNA in the second (turn it around?).
        mirNames (list): Names of the microRNAs relevant for the construction of the graph.
        geneNames (list): Names of the genes to use for the construction of the graph.

    Returns:

        globalGraph (nx.DiGraph): Returns the information of the micro-gen relationships in directed graph format from the Networx package (DiGraph()).

    """

    # Initialize the graph.

    globalGraph = nx.DiGraph()
    TScanPredict = inputFile
    mirNames = [mir.lower() for mir in mirNames]

    for mir in mirNames:
        globalGraph.add_node(mir.lower())

    for gene in geneNames:
        globalGraph.add_node(gene)

    print('LOADING ENGINE...')

    mirNames_set = set(mirNames)
    geneNames_set = set(geneNames)

    for tail, head in tqdm(zip(TScanPredict.iloc[:,0], TScanPredict.iloc[:,1].str.lower()), total=len(TScanPredict)):
        if head not in mirNames_set:
            continue
        if tail not in geneNames_set:
            continue

        if globalGraph.has_edge(tail, head):
            globalGraph[tail][head]['weight'] += 1
        else:
            globalGraph.add_edge(tail, head, weight=1)

    # print('\nGRAPH INITIALIZED')
    # print('NUMBER OF NODES: ', globalGraph.number_of_nodes())
    # print('NUMBER OF EDGES: ', globalGraph.number_of_edges(), '\n')

    return globalGraph
    






def run_engine_scikit(X_train, y_train):

    """
    
    Initializes the structural engine from a train dataset. It filters the miRNAs and mRNAs depending on the databases of predicted interactions TargetScan, DIANA and miRTarBase.
    
    Args:
        X_train (pd.DataFrame): Input dataset with expression value of predictive variables (miRNAs & mRNAs).
        y_train (pd.Series): Binary input dataset with target variable (class or phenotype).
    
    Returns:
        structural_information (np.ndarray): Structural information between each pair of nodes. The matrix will be of dimension (n+m)*(n+m) being n = number of microRNAs and m = number of genes.
        micros: List with the names of the microRNAs of the model. Those that are in the input database and in the structural engine.
        genes: List with the names of the genes of the model. Those that are in the input database and in the structural engine.
        conex (int): Number of potential connections.
        gtf: File with gene annotations.

    
    """

    print('STARTING ENGINES...')

    # Read the gtf file.
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # gtf_path = os.path.join(current_dir, "StructuralEngine/gencode.v47.basic.annotation.gtf")
    # gtf = read_gtf(gtf_path)

    gtf_path = os.path.join(current_dir, 'engines', 'gencode.v47.basic.annotation.gtf')
    gtf = pd.read_csv(
        #'/rnacorex/engines/gencode.v47.basic.annotation.gtf',
        gtf_path,
        sep='\t',
        comment='#',
        header=None,
        names=[
            'seqname', 'source', 'feature', 'start', 'end',
            'score', 'strand', 'frame', 'attribute'
        ]
    )

    gtf['gene_id'] = gtf['attribute'].str.extract(r'gene_id "([^"]+)"')
    gtf['gene_name'] = gtf['attribute'].str.extract(r'gene_name "([^"]+)"')
    gtf = gtf[['gene_id', 'gene_name']]
    mapping_dict = dict(zip(gtf['gene_name'], gtf['gene_id']))

    # Read the databases of predicted interactions.

    diana_path = os.path.join(current_dir, 'engines', 'DIANA_targets.txt')
    tscan_path = os.path.join(current_dir, 'engines', 'Targetscan_targets.txt')
    mtbase_path = os.path.join(current_dir, 'engines', 'MTB_targets_25.csv')
    tarbase_path = os.path.join(current_dir, 'engines', 'TarBase_v9.tsv')

    diana = pd.read_csv(diana_path, sep='\t')
    targetscan = pd.read_csv(tscan_path, sep='\t')
    mirtarbase = pd.read_csv(mtbase_path)
    tarbase = pd.read_csv(tarbase_path, sep='\t')

    tarbase = tarbase[tarbase['mirna_name'].str.contains('hsa')]
    tarbase = tarbase[['gene_id', 'mirna_name']]
    tarbase.columns = ['Gene ID', 'miRNA']
    tarbase = tarbase[tarbase['miRNA'].str.lower().str.startswith('hsa')]
    tarbase['miRNA'] = tarbase['miRNA'].str.lower()
    tarbase.drop_duplicates(inplace=True)

    # Filter the databases to keep only the columns of interest (gene and miRNA).

    diana = diana.loc[:, ['ensembl_gene_id', 'mirna']]
    targetscan = targetscan.loc[:, ['Gene ID', 'miRNA']]
    mirtarbase = mirtarbase.loc[:, ['Target Gene', 'miRNA']] 
    mirtarbase['Target Gene'] = mirtarbase['Target Gene'].str.upper()

    # Clean databases to get every mRNA in Ensemble format and every miRNA in miRBase nomenclature and lowercase.

    mirtarbase['Target Gene'] = mirtarbase['Target Gene'].apply(lambda x: mapping_dict.get(x, x))

    targetscan['Gene ID'] = [s.split('.')[0] for s in targetscan['Gene ID']]
    diana = diana[diana['mirna'].str.startswith('hsa')]
    targetscan = targetscan.dropna()
    targetscan = targetscan[targetscan['miRNA'].str.startswith('hsa')]
    targetscan['miRNA'] = targetscan['miRNA'].str.lower()
    diana['mirna'] = diana['mirna'].str.lower()
    mirtarbase['Target Gene'] = [s.split('.')[0] for s in mirtarbase['Target Gene']]
    mirtarbase['miRNA'] = mirtarbase['miRNA'].str.lower()

    # Get unique genes and miRNAs from the databases.

    gen_target = targetscan['Gene ID']
    gen_DIANA = diana['ensembl_gene_id']
    gen_mirtarbase = mirtarbase['Target Gene']
    gen_tarbase = tarbase['Gene ID']
    micros_target = targetscan['miRNA']
    micros_DIANA = diana['mirna']
    micros_mirtarbase = mirtarbase['miRNA']
    micros_tarbase = tarbase['miRNA']

    gen_target = set(gen_target)
    gen_DIANA = set(gen_DIANA)
    gen_mirtarbase = set(gen_mirtarbase)
    gen_tarbase = set(gen_tarbase)
    micros_target = set(micros_target)
    micros_DIANA = set(micros_DIANA)
    micros_mirtarbase = set(micros_mirtarbase)
    micros_tarbase = set(micros_tarbase)

    # Join all the genes and miRNAs from the databases.

    genes_str = gen_target.union(gen_DIANA).union(gen_mirtarbase).union(gen_tarbase)
    micros_str = micros_target.union(micros_DIANA).union(micros_mirtarbase).union(micros_tarbase)

    genes_str = list(genes_str)
    micros_str = list(micros_str)

    genes_str = sorted(genes_str)
    micros_str = sorted(micros_str)

    # Get the names of the genes and miRNAs from the input dataset.

    cols = list(X_train.columns.values)
    geneNames = [element for element in cols if element.startswith('ENSG')]
    mirNames = [element for element in cols if element.startswith('hsa')]

    # Filter the genes and miRNAs from the input dataset to keep only those that are in the databases.

    micros = list(set(micros_str) & set(mirNames))
    genes = list(set(genes_str) & set(geneNames))

    micros = sorted(micros)
    genes = sorted(genes)

    micros_test = micros.copy()
    genes_test = genes.copy()
    micros_test.append('classvalues')
    genes_test.append('classvalues')

    # Initialize graphs.

    TargetScanGraph = initialize_graph(targetscan, micros, genes)

    DIANAGraph = initialize_graph(diana, micros, genes)

    MIRTARGraph = initialize_graph(mirtarbase, micros, genes)

    TARBASEGraph = initialize_graph(tarbase, micros, genes)
    
    # Join the three graphs in a unique list.

    glist = [TargetScanGraph, DIANAGraph, MIRTARGraph, TARBASEGraph]

    # Calculate the structural information matrix.

    structural_information = ensemble_graph(glist, len(micros)*len(genes), consistency_scale = 2)

    # Print the information of the model.

    print('\nMODEL INITIALIZED')
    print('MicroRNAs: ', len(micros))
    print('Genes: ', len(genes))
    print('Connections: ', len(structural_information.nonzero()[0]))

    conex = len(structural_information.nonzero()[0])
    
    return structural_information, micros, genes, conex, gtf