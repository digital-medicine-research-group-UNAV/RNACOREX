import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import sys
import os
from auxiliary_functions import get_genes, benjamini_hochberg, rename_all
from gtfparse import read_gtf

"""

FUNCIONALIDAD

Dados dos grafos con n microRNAs y m genes, la función devuelve la intersección de las matrices de adyacencia (enlaces que coinciden en ambos grafos).
La matriz es dicotómica (True/False) en la que un elemento es True si la arista es común en ambos grafos.

INPUTS

g1: Grafo dirigido Digraph() del paquete Networkx.
g2: Grafo dirigido Digraph() del paquete Networkx.

g1 y g2 tienen que estar compuestos por los mismos nodos.

OUTPUTS

intersec_matrix: Matriz dicotómica (True/False) de dimensión (n+m)*(n+m) en la que un elemento es True si la arista es común en ambos grafos.

No tiene en cuenta el peso de las conexiones.

"""

def intersection_adjmatrix(g1, g2):

    # Calcula la matriz de adyacencias del primer grafo
    adjmatrix1 = nx.adjacency_matrix(g1).todense()
    adjmatrix1 = adjmatrix1 > 0
    adjmatrix1 = np.maximum(adjmatrix1, adjmatrix1.T)

    # Calcula la matriz de adyacencias del segundo grafo
    adjmatrix2 = nx.adjacency_matrix(g2).todense()
    adjmatrix2 = adjmatrix2 > 0
    adjmatrix2 = np.maximum(adjmatrix2, adjmatrix2.T)

    # Calcula la intersección de matrices
    intersec_matrix = adjmatrix1 & adjmatrix2

    # Convierte la matriz en dicotómica (True/False)
    intersec_matrix = intersec_matrix > 0

    return intersec_matrix





"""

FUNCIONALIDAD

Función que devuelve el índice de consistencia entre dos grafos. El índice será mayor cuanto mayor coincidencia haya entre grafos.

- R. Armañanzas. Revealing post-transcriptional microRNA-mRNA regulations in Alzheimer's disease through ensemble graphs. 
BMC Genomics. 2018 Sep 24;19(Suppl 7):668. doi: 10.1186/s12864-018-5025-y. PMID: 30255799; PMCID: PMC6157163.


INPUTS

g1: Grafo dirigido Digraph() del paquete Networkx.
g2: Grafo dirigido Digraph() del paquete Networkx.
n: Número máximo de posibles conexiones. Si sólamente hay conexiones micro-gen: nº micros * nº genes.
scale: Tipo de escalado que se quiere aplicar.
        0: Sin escalar
        1: Escala el índice de [-km,1] a [0,1]
        2: Escala el índice de [-1,1] a [0,1]


OUTPUTS

ci: Valor real del índice de consistencia

"""

def consistency_index(g1, g2, n, scale=0):

    k1 = g1.number_of_edges()
    k2 = g2.number_of_edges()

    intersecadjmatrix = intersection_adjmatrix(g1,g2)

    # Aristas comunes
    r = (intersecadjmatrix.sum())/2

    # Máximo número de aristas entre los dos grafos
    km = np.maximum(k1,k2)

    # Índice de consistencia (calcula la similaridad entre los grafos según sus aristas)
    ci_top = r*n-km*km
    nkm = (n-km)*1.0
    ci_down = km*nkm
    ci = ci_top / ci_down

    if scale == 1: # Escala el índice de [-km,1] a [0,1]
        ci = (ci+km)/(1+km)

    if scale == 2: # Escala el índice de [-1,1] a [0,1]
        if ci < -1:
            # ci = -1
            ci = 0 # NO??
        else:
            km = 1
            ci = (ci+km) / (1+km)

    return ci

"""

FUNCIONALIDAD

ensemble_graph() calcula la matriz de información mutua estructural a partir de un conjunto de grafos que se le pasan a la función.
Los grafos se ponderan utilizando el resultado obtenido en su índice de consistencia.
Es imprescindible que los grafos que se le pasen a la función estén formados por los mismos nodos y por tanto TENGAN LA MISMA DIMENSIÓN.


INPUTS

glist: Lista de n DAGs (networx Digraph()) todos ellos con los mismos nodos.
c: Número máximo de posibles conexiones. Si sólamente hay conexiones micro-gen: nº micros * nº genes.
consistency_scale: Tipo de escalado que se quiere aplicar.


OUTPUTS

structural_information: Matriz de numpy de dimensión n_nodos x n_nodos conteniendo la información mutua estructural de cada par de nodos.

"""


def ensemble_graph(glist, c, consistency_scale = 0):

    # Nº de grafos
    n_graphs = len(glist)

    # Inicialización de la matriz de ponderación de cada grafo
    wmatrix = np.zeros((n_graphs, n_graphs))

    # Calculo del índice de consistencia entre cada par de grafos
    for i in range(0, n_graphs-1):
        gi = glist[i]

        for j in range((i+1), n_graphs):
            gj = glist[j]

            # Se calcula el índice de consistencia para cada par de grafos
            w = consistency_index(gi, gj, c, scale = consistency_scale)

            # Se asigna el índice de consistencia a la matriz de ponderación
            wmatrix[i][j] = w

    # Se relativizan los índices de consistencia de forma que todos sumen 1 y cada elemento esté en el rango [0,1]
    wmatrix = wmatrix/np.sum(wmatrix)
    wmatrix[np.isnan(wmatrix)] = 0

    # Se genera la matriz para la información funcional
    n_nodes = glist[0].number_of_nodes()
    edgewam = np.zeros((n_nodes, n_nodes))

    # Se va escogiendo cada par de grafos
    for i in range(0, n_graphs-1):
        gi = glist[i]

        for j in range((i+1), n_graphs):
            gj = glist[j]

            # Se calcula la intersección entre ambos grafos (aristas que están en ambos y con el peso mínimo).
            intersectionadjmatrix = intersection_adjmatrix(gi,gj)

            # A la matriz final de pesos de las conexiones se le añade el valor obtenido arriba multiplicado por el índice de consistencia.
            edgewam = edgewam + intersectionadjmatrix * wmatrix[i][j]
    
    # Genera un dataframe con la información mutua de forma que en los índices y las columnas estén los nombres de los nodos (microRNAs y genes)
    structural_information = np.triu(edgewam)

    return structural_information

import networkx as nx
from tqdm import tqdm
import csv


""" 

FUNCIONALIDAD

La función construye un grafo dirigido (DAG) con las relaciones estructurales de un formato .csv.
A se le pueden indicar aquellos genes y microRNA-s que se consideran relevantes para el estudio a través de dos .csv-s (mirNames y geneNames).
Si no se indican genes y microRNA-s por defecto inicializa el grafo con todas las conexiones del archivo.
En caso de que se indiquen microRNA-s y genes, la función ignora todos los nodos que no estén incluidos en dichos archivos.
Si sólo se incluyen los nombres de uno de los dos elementos la función te pide que incluyas los otros o no incluyas ninguno ''


INPUTS

inputFile: Archivo .csv con la información de las relaciones micro-gen. Cada línea tiene que ser una relación, teniendo en la primera columna el gen y en la segunda el micro RNA (darle la vuelta?).
mirNames: Nombres de los microRNAs relevantes para la construcción del grafo.
geneNames: Nombre de los genes a utilizar para la construcción del grafo.


OUTPUTS

globalGraph: Devuelve la información de las relaciones micro-gen en formato grafo dirigido del paquete Networx (DiGraph()).

"""


def initialize_graph(inputFile, mirNames = None, geneNames = None):

    # Inicializar el grafo.

    globalGraph = nx.DiGraph()
    TScanPredict = inputFile
    mirNames = [mir.lower() for mir in mirNames]

    for mir in mirNames:
        globalGraph.add_node(mir.lower())

    for gene in geneNames:
        globalGraph.add_node(gene)

    # Se inicializan las aristas / ejes. El peso de cada arista es el número de veces que aparece cada conexión.

    print('INICIALIZANDO GRAFO...')

    for i in tqdm(range(0,len(TScanPredict))):

        tail = TScanPredict.iloc[i,0]
        head = TScanPredict.iloc[i,1].lower()

        # Si tanto la primera columna (gen) como la segunda (microRNA) coinciden con aquellos vértices que se han añadido al grafo se inicializa una arista entre ellos.

        if head not in mirNames:

            continue
        
        if tail in geneNames:

            # En caso de que el eje ya exista se aumenta en 1 su ponderación.

            if globalGraph.has_edge(tail, head):
                globalGraph[tail][head]['weight'] += 1

            # Si no existe se crea el eje.

            else:
                globalGraph.add_edge(tail, head, weight=1)

    # Cuando termina de inicializar el grafo lo muestra por pantalla junto al número de nodos y aristas del mismo.

    print('\nGRAPH INITIALIZED')
    print('NUMBER OF NODES: ', globalGraph.number_of_nodes())
    print('NUMBER OF EDGES: ', globalGraph.number_of_edges(), '\n')
    

    return globalGraph


"""

FUNCIONALIDAD

La función , a partir de un par de listas con los nombres de los miRNA y los genes de entrada filtra los elementos a únicamente aquellos que se encuentran en el motor estructural.
Una vez definidos dichos nodos inicializa los grafos y los ensambla. Finalment, calcula la información estructural entre cada par microRNA-gen y lo devuelve como matriz de numpy.
Para ello hace uso de las funciones initialize_graph() y ensemble_graph().


INPUTS

mirNames: Lista con los nombres de los microRNA de la base de datos de entrada.
geneNames: Lista con los nombres de los genes de la base de datos de entrada.


OUTPUTS

structural_information: Matriz de numpy con la información estructural entre cada par de nodos. La matriz será de dimensión (n+m)*(n+m) siendo n = nº de microRNA y m = nº de genes.
micros: Listado con los nombres de los microRNA del modelo. Aquellos que se encuentran en la base de datos de entrada y en el motor estructural.
genes: Listado con los nombres de los genes del modelo. Aquellos que se encuentran en la base de datos de entrada y en el motor estructural.


"""


def run_engine(datos_original = None, datos_test = None, alpha = 0.05, num_genes = 0, save = 'url', ignore_sign = True, scaled = False, link_txt=False):

    print('INICIALIZANDO MOTORES...')

    if link_txt != False:
        with open(link_txt, 'w') as f:
            sys.stdout = f
            print('Patients: ', datos_original.shape[0])
            print(f"Class distribution: {np.bincount(datos_original['classvalues'])}")
            sys.stdout = sys.__stdout__


    gtf = read_gtf("Otros/gencode.v45.chr_patch_hapl_scaff.basic.annotation.gtf")

    diana = pd.read_csv('Motor_Estructural/Motores_Aitor/DIANA_targets.txt', sep='\t')
    targetscan = pd.read_csv('Motor_Estructural/Motores_Aitor/Targetscan_targets.txt', sep='\t')
    mirtarbase = pd.read_csv('Motor_Estructural/Motores_Aitor/MTB_targets.csv', sep=';')

    diana = diana.loc[:, ['ensembl_gene_id', 'mirna']]
    targetscan = targetscan.loc[:, ['Gene ID', 'miRNA']]
    mirtarbase = mirtarbase.loc[:, ['Target Gene', 'miRNA']]

    map = rename_all(mirtarbase['Target Gene'], gtf)
    mirtarbase['Target Gene'] = map

    targetscan['Gene ID'] = [s.split('.')[0] for s in targetscan['Gene ID']]
    diana = diana[diana['mirna'].str.startswith('hsa')]
    targetscan = targetscan.dropna()
    targetscan = targetscan[targetscan['miRNA'].str.startswith('hsa')]
    targetscan['miRNA'] = targetscan['miRNA'].str.lower()
    diana['mirna'] = diana['mirna'].str.lower()
    mirtarbase['miRNA'] = mirtarbase['miRNA'].str.lower()

    gen_target = targetscan['Gene ID']
    gen_DIANA = diana['ensembl_gene_id']
    gen_mirtarbase = mirtarbase['Target Gene']
    micros_target = targetscan['miRNA']
    micros_DIANA = diana['mirna']
    micros_mirtarbase = mirtarbase['miRNA']

    gen_target = set(gen_target)
    gen_DIANA = set(gen_DIANA)
    gen_mirtarbase = set(gen_mirtarbase)
    micros_target = set(micros_target)
    micros_DIANA = set(micros_DIANA)
    micros_mirtarbase = set(micros_mirtarbase)

    genes_str = gen_target.union(gen_DIANA).union(gen_mirtarbase)
    micros_str = micros_target.union(micros_DIANA).union(micros_mirtarbase)

    genes_str = list(genes_str)
    micros_str = list(micros_str)

    genes_str = sorted(genes_str)
    micros_str = sorted(micros_str)

    cols = list(datos_original.columns.values)
    geneNames = [element for element in cols if element.startswith('ENSG')]
    mirNames = [element for element in cols if element.startswith('hsa')]

    micros = list(set(micros_str) & set(mirNames))
    genes = list(set(genes_str) & set(geneNames))

    micros = sorted(micros)
    genes = sorted(genes)

    micros_test = micros.copy()
    genes_test = genes.copy()
    micros_test.append('classvalues')
    genes_test.append('classvalues')

    if ignore_sign == False:

        pvalues_corrected_g, p_values_g = benjamini_hochberg(datos_original[genes_test])

        genes = get_genes(p_values_g, alpha = alpha, number_genes = num_genes)
    
        genes = genes.to_list()

    TargetScanGraph = initialize_graph(targetscan, micros, genes)

    if link_txt != False:

        with open(link_txt, 'a') as f:
            sys.stdout = f
            print('\nTARGETSCAN')
            print('NUMBER OF NODES: ', TargetScanGraph.number_of_nodes())
            print('NUMBER OF EDGES: ', TargetScanGraph.number_of_edges(), '\n')
            sys.stdout = sys.__stdout__

    DIANAGraph = initialize_graph(diana, micros, genes)

    if link_txt != False:

        with open(link_txt, 'a') as f:
            sys.stdout = f
            print('\nDIANA')
            print('NUMBER OF NODES: ', DIANAGraph.number_of_nodes())
            print('NUMBER OF EDGES: ', DIANAGraph.number_of_edges(), '\n')
            sys.stdout = sys.__stdout__

    MIRTARGraph = initialize_graph(mirtarbase, micros, genes)

    if link_txt != False:

        with open(link_txt, 'a') as f:
            sys.stdout = f
            print('\nMIRTAR')
            print('NUMBER OF NODES: ', MIRTARGraph.number_of_nodes())
            print('NUMBER OF EDGES: ', MIRTARGraph.number_of_edges(), '\n')
            sys.stdout = sys.__stdout__

    glist = [TargetScanGraph, DIANAGraph, MIRTARGraph]

    if scaled == True:

        datos_scale = datos_original.drop(datos_original.columns[-1], axis=1)
        datos_scale = datos_scale[micros+genes]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler() 
        scaled_values = scaler.fit_transform(datos_scale) 
        datos_scale.loc[:,:] = scaled_values

        dataset = {"data": datos_scale, "classvalues": datos_original.iloc[:, -1]}
    
    else:

        dataset = {"data": datos_original[micros+genes], "classvalues": datos_original.iloc[:, -1]}
    
    
    if datos_test is None:
        
        print('Sin test')
    
    else:
        
        dataset_test = {"data": datos_test[micros+genes], "classvalues": datos_test.iloc[:, -1]}


    structural_information = ensemble_graph(glist, len(micros)*len(genes), consistency_scale = 2)

    print('\nMODEL INITIALIZED')
    print('MicroRNAs: ', len(micros))
    print('Genes: ', len(genes))
    print('Connections: ', len(structural_information.nonzero()[0]))

    if link_txt != False:

        with open(link_txt, 'a') as f:
            sys.stdout = f
            print('\nMODEL INITIALIZED')
            print('MicroRNAs: ', len(micros))
            print('Genes: ', len(genes))
            print('Connections: ', len(structural_information.nonzero()[0]))
            sys.stdout = sys.__stdout__

    if save != 'url':

        structural_info = pd.DataFrame(structural_information)
        structural_info.to_csv(save)
        dir_path = os.path.dirname(save)
        nx.write_graphml(TargetScanGraph, dir_path+'/TargetScanGraph.graphml')
        nx.write_graphml(DIANAGraph, dir_path+'/DIANAGraph.graphml')
        nx.write_graphml(MIRTARGraph, dir_path+'/MIRTARGraph.graphml')

    if datos_test is None:
        
        return structural_information, micros, genes, dataset
    
    else:

        return structural_information, micros, genes, dataset, dataset_test