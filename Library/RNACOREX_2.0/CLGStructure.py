import numpy as np
from scipy.stats import norm
from scipy.linalg import inv
import networkx as nx
import pandas as pd
from tqdm import tqdm


def fit_model(X, y, smi, fmi, n_con, info):

    # Construye el DAG con las principales interacciones según SMI y FMI.

    binary_matrix, coordinates = build_dag(smi, fmi, n_con)

    # Filtra el DAG, la info y los datos de expresión quedándose únicamente con aquellos que se utilizan en el modelo.

    filtered_dag, filtered_fmi, filtered_smi, info, X, classes = filter_dag(X, y, smi, fmi, binary_matrix, coordinates, info)

    # Estima los parámetros del modelo.

    clgc = pl_plemclg(X, classes, filtered_dag, filtered_fmi, filtered_smi, info)

    clgc = get_network(clgc)

    return clgc




def pl_plemclg(dataset, classpl, dag, fmi, smi, info):

    """
    
    The structure_search() function executes the pl_plemclg() function for each network to create. The pl_plemclg() function redirects to the clg_param_continuous() 
    and pl_factorizedpdf() functions to calculate the model parameters and the probabilities of each individual. With the information it receives, 
    it calculates the metrics of the DAG (acc, ll, auc) and returns them to structure_search().

    Args:

        dataset (numpy.ndarray): Matrix with the expression data of the nodes to include in the DAG.
        classpl (numpy.ndarray): Matrix with k columns (as many as classes) in which each element in the column will be a 1 if the element belongs to that class and 0 if not.
        dag (numpy.ndarray): Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.

    Returns:

        rl (dict): Dictionary with the model parameters and other information.
            1) 'classp': Prior probability of each class.
            2) 'beta': Matrix nxn (being n the number of nodes in the DAG) with the betas of the model.
            3) 'sigma': Array with the sigmas of the model.
            4) 'dag': Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
            5) 'll': Log-likelihood of the DAG.
            6) 'acc': DAG accuracy.

    """


    m = dataset.shape[0]
    n = dataset.shape[1]
    k = classpl.shape[1]

    classp = np.zeros(k)
    sigma = np.zeros((n,k))
    beta = np.zeros((n,n,k))

    pdf = np.zeros((m,k))

    # Para cada clase

    for c in range(0,k):
            
        # Calcular la probabilidad a priori.

        classp[c] = np.sum(classpl[:,c], axis=0)
        classp[c] /= m

        # Estimar los parámetros.

        variance, betas = clg_param_continuous(classpl[:,c], dataset, dag)

        sigma[:, c] = np.transpose(variance).ravel()
        
        beta[:, :, c] = betas

        # Obtener probabilidades de cada muestra para cada clase.

        pdf[:,c] = pl_factorizedpdf(dataset,beta[:,:,c], sigma[:,c], dag)

    replicated_classp = np.tile(classp, (m, 1))

    # classpl is a vector with k columns that in each column shows a 1 if that element belongs to that class and 0 if not.
    # replicated_classp repeats the prior probability of each class m times (number of columns / nodes).
    # pdf is the probability associated with each element and each class.
    # It is multiplied by the prior probability (only the probability of its class, the other is canceled by *0)

    posterior_prob = replicated_classp * pdf

    # Max posterior probability is selected as the predicted class for each sample.

    maxclass = np.argmax(posterior_prob, axis=1)

    # The second column is taken because the binary vector has been reversed so that the 1s correspond to each class, therefore the original classes are in column 1.

    # Pct of coincidences between predicted and real classes are used for calculating accuracy.

    accuracy = np.sum(classpl[:,1] == maxclass)/len(classpl[:,1])

    # We take the probability of the real class.

    prob_ll = classpl * posterior_prob

    # Sum all probabilities for real classes.

    sumprob = np.sum(prob_ll, axis=1)

    ll = np.sum(np.log(sumprob))

    bic = ll-(np.log(n)/2)*dimension(dag,k)

    rl = {"classp":classp,"beta":beta,"sigma":sigma,"dag":dag,"labels":info["labels"],"types":info["types"],"names":info["names"],"fmi":fmi,"smi":smi,"ll":ll,"acc":accuracy,"bic":bic}

    return rl


def clg_param_continuous(classes, dataset, dag):

    """

    The clg_param_continuous() function calculates the parameters of the model (betas and sigmas) for a given dataset and a given DAG.

    Args:

        classes(np.ndarray): Binary array (0/1) that indicates if an element belongs to the class (1) or not (0). The function only takes into account those elements that do belong.
        dataset(np.ndarray): Matrix with the expression values of the individuals in the microRNA and genes that are present in the DAG.
        dag(np.ndarray): Binary matrix (n x n), where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.

    Returns:

        variance (np.ndarray): Array with the variances of the nodes.
        betamat (np.ndarray): Matrix with the values of the betas of the model.

    """

    MIN_DOUBLE = 4.9406564584124654e-50

    # Number of individuals.

    m = dataset.shape[0]

    # Number of nodes (mRNAs + miRNAs).

    n = dataset.shape[1]

    N = classes.sum()

    mu = np.zeros((1,n)) 
    variance = np.zeros((1,n)) 
    diffvar = np.zeros((m,n)) 
    covmat = np.zeros((n,n))
    betamat = np.zeros((n,n))

    # Para cada nodo se calcula su valor de expresión medio en la clase (mu), la diferencia de cada valor de expresión con el valor medio de ese nodo (diffvar) y la diagonal de la matriz de covarianza (covmat).

    for i in range(0,n):
        mu[0,i] = np.sum(classes*dataset.iloc[:,i]) / N
        diffvar[:,i] = dataset.iloc[:,i] - mu[0][i]
        covmat[i,i] = np.sum(classes*diffvar[:,i]*diffvar[:,i]) / N 

    # Se añaden a la matriz de covarianza los elementos que no están en la diagonal.

    for i in range(0,n):
        for j in range(0,n):
            if i != j:
                covmat[i,j] = np.sum(classes*diffvar[:,i]*diffvar[:,j]) / N

    # Se calculan los parámetros del modelo para cada nodo.

    for i in range(0,n):

        # Se seleccionan los padres del nodo i.

        pa = np.where(dag[:, i] == 1)[0]
        npa = len(pa)

        # Si no tiene padres, es un miRNA, la media es el valor medio de expresión del nodo y la varianza es la de la diagonal de la matriz de covarianza.

        if npa == 0:
            betamat[i,i] = mu[0][i]
            variance[0][i] = covmat[i,i]

        # Si tiene padres, se calculan los parámetros del modelo. Este sería el caso de los mRNA.

        else:

            for parentindex in range(0, npa):

                j = pa[parentindex]
                covipa = covmat[i,pa]
                covpai = covmat[pa,i]
                covii = covmat[i,i]
                covpapa = covmat[np.ix_(pa, pa)]

            if npa == 1:
                div = 1 / covpapa
                betamat[i,i] = mu[0][i] - covipa * div * mu[0][pa] # Beta0
                betamatipa = div * covpai
                betamat[i, pa] = betamatipa.reshape(npa)
                variance[0][i] = covii - covipa * div * covpai
                
            else:
                            
                # FOR SINGULAR MATRICES
                            
                if np.linalg.det(covpapa) == 0:
                    Q, R = np.linalg.qr(covpapa)
                    covpapa_inv = np.linalg.solve(R, Q.T)
                    betamat[i,i] = mu[0][i] - covipa @ covpapa_inv @ mu[0][pa]
                    betamatipa = covpapa_inv @ covpai
                    betamat[i, pa] = betamatipa.reshape(npa)
                    variance[0][i] = covii - covipa @ covpapa_inv @ covpai
                    print('NODO', i)
                    print('SINGULAR')

                else:

                    betamat[i,i] = mu[0][i] - covipa @ inv(covpapa) @ mu[0][pa]   
                    betamatipa = inv(covpapa) @ covpai
                    betamat[i, pa] = betamatipa.reshape(npa)
                    variance[0][i] = covii - covipa @ inv(covpapa) @ covpai

    return variance, betamat


def pl_factorizedpdf(data, beta, sigma, dag):

    
    """

    Given some data and the parameters of a CLG model, pl_factorizedpdf() calculates the probability associated with each individual.
    
    Args:

        data (np.ndarray): Matrix with the expression values of the individuals in the microRNA and genes that are present in the DAG.
        beta (np.ndarray): Matrix with the values of the betas of the model.
        sigma (np.ndarray): Array with the sigmas of the model.
        dag (np.ndarray): Binary matrix (n x n), where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.

    Returns:

        pdf (np.ndarray): Returns the probability associated with each individual according to the DAG.

    """

    # Nº of individuals
    m = data.shape[0]

    # Nº of nodes
    n = dag.shape[1]
    pdf = np.ones(m)

    # Convert variances into deviations.
    stdevs = np.sqrt(sigma)

    # For each DAG node
    for i in range(n):

        # Find parents
        pa = np.where(dag[:,i] == 1)[0]

        # If it has no parents :(
        if len(pa) == 0:
           
            # The probability depends only on the values of the node itself (its mean and deviation).
            # Se identifica la probabilidad de cada expresión (nº individuos) en base a su distribución (1 x nindividuos).
            # Identify the probability of each expression (nº individuals) based on its distribution (1 x nindividuals).
            pdfaux = norm.pdf(data.iloc[:,i], loc=beta[i][i], scale=stdevs[i])
        
        # If it has parents :)
        else:

            # Check how many parents it has
            npa = len(pa)

            # Repeat its deviation m times (1 for each individual)
            stdev = np.repeat(stdevs[i], m)

            # Repeat m times the Beta0 of the element
            mu = np.repeat(beta[i][i], m)

            # Join the indices of the parents with the position of the corresponding node repeated npa times (to obtain the indices with which to search in the parameter matrix)
            replicated_i = np.repeat(i, npa)
            ind = np.column_stack((replicated_i, pa))

            # Obtain with the beta matrix the parameters of the distributions of the parents (the betas)
            result = beta[ind[:, 0], ind[:, 1]]

            # Repeat as many times as there are data
            replicated_beta = np.tile(result, (m, 1))

            # Multiply the betas by the expression values of the data
            product_matrix = replicated_beta * data.iloc[:, pa]
            sum_vector = np.sum(product_matrix, axis=1)
            
            # Sum beta0
            mu += sum_vector

            # Obtain the associated probability in the distribution
            pdfaux = norm.pdf(data.iloc[:,i], loc=mu, scale=stdev)

        pdf = pdf*pdfaux

    return pdf


# AUXILIARY FUNCTIONS

def build_dag(smi, fmi, n_con):

    SMI_flat = smi.flatten()
    FMI_flat = fmi.flatten()

    n_rows, n_cols = smi.shape

    selected_indices = []

    selected_set = set()

    # Selection alternated between SMI and FMI
    for i in range(n_con):  
        if i % 2 == 0 and n_con > 0:  
            max_smi_index = np.argmax(SMI_flat)
            while max_smi_index in selected_set:
                SMI_flat[max_smi_index] = -np.inf  
                max_smi_index = np.argmax(SMI_flat)
            selected_indices.append(max_smi_index)
            selected_set.add(max_smi_index)
            n_con -= 1  
        elif i % 2 == 1 and n_con > 0:  
            max_fmi_index = np.argmax(FMI_flat)
            while max_fmi_index in selected_set:
                FMI_flat[max_fmi_index] = -np.inf  
                max_fmi_index = np.argmax(FMI_flat)
            selected_indices.append(max_fmi_index)
            selected_set.add(max_fmi_index)
            n_con -= 1  

    # Convertir los índices seleccionados en la matriz binaria
    binary_matrix = np.zeros((n_rows, n_cols), dtype=int)
    coordinates = []

    # Asignar 1 a los índices seleccionados
    for idx in selected_indices:
        row, col = np.unravel_index(idx, (n_rows, n_cols))
        binary_matrix[row, col] = 1
        coordinates.append([row, col])


    return binary_matrix, coordinates





def filter_dag(X, y, smi, fmi, dag, coordinates, info):

    # Filter DAG

    flattened = [item for sublist in coordinates for item in sublist]
    flattened = np.unique(flattened)
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

    ndata = len(X)

    k = np.max(y) + 1
    k = int(k)

    classes = np.zeros((ndata, k))

    for i in range(0, ndata):
        for j in range(0,k):
            if y.iloc[i] == j:
                classes[i][j] = 1
    
    return filtered_dag, filtered_fmi, filtered_smi, info, X_filtered, classes


def dimension(dag, k):

    """

    Calculates the dimension of the DAG according to the number of nodes and the number of classes. It is used specifically for the BIC correction.

    Args:

        dag (np.ndarray): Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
        k (int): Number of classes (phenotypes).

    Returns:

        dim (int): Dimension of the model.

    """

    n = dag.shape[1]
    dim = k-1

    for i in range(n):
        npa = np.sum(dag[:, i])
        dim += npa * k + 2 * k 

    return dim



def get_network(clgc):

    G = nx.MultiDiGraph()

    color_map = {

        "miRNA": "lightgreen", 
        "mRNA": "skyblue",
        "lncRNA": "lightcoral",
        "default": "gray"
    }

    for i, node in enumerate(clgc["names"]):
        node_type = clgc["types"][i] 
        color = color_map.get(node_type, color_map["default"]) 
        G.add_node(node, color=color)

    # Iterate over all rows and columns in the binary matrix
    for i in range(clgc['dag'].shape[0]):  # Row names
        for j in range(clgc['dag'].shape[1]):  # Column names
            if clgc['dag'][i, j] == 1:  # If there's a 1 in the matrix
                G.add_edge(clgc["names"][i], clgc["names"][j])  # Add edge from row_node to col_node

    clgc["G"] = G
    
    return clgc


def predict_test(X_test, clgc):

    classifications = []
    posteriors = []
    claves = []

    for clave in tqdm(clgc, desc="Predicting", unit="model"):

        model = clgc[clave]
     
        dataset_te = X_test[model['labels']]

        m = X_test.shape[0]

        k = len(model['classp'])

        condpdfx = np.zeros((m,k))
        
        for c in range(0,k):

            condpdfx[:,c] = pl_factorizedpdf(dataset_te, model['beta'][:,:,c], model['sigma'][:,c], model['dag'])

        priorprob = model["classp"]

        posterior = np.zeros((m,k))

        for c in range(0,k):

            posterior[:,c] = priorprob[c]*condpdfx[:,c]

        # m a posteriori probabilities are obtained for each class, one for each individual. It is the probability associated with each class for each individual.

        # tsum includes the sum of the probabilities associated with each class.
        tsum = np.sum(posterior,axis=1)

        # We only keep the positive sums.
        selec = tsum > 0

        # Relativizes the values of each class.
        posterior[selec, :] /= tsum[selec, np.newaxis]

        # Equalizes negative probabilities to 0.
        posterior[~selec, :] = 0

        # In cases where there was a negative probability (which has been assigned to 0), the other is assigned to 1.
        posterior[~selec, np.argmax(priorprob)] = 1

        # The column with the highest probability is searched for and that is used as the class that is predicted for the instance.
        maxclass = np.argmax(posterior, axis=1)

        posteriors.append(posterior)

        classifications.append(maxclass)

        claves.append(clave)

    rl = {"posterior":posteriors, "classification":classifications, "k":claves}

    return rl


def get_interactions(smi, fmi, info, n):

    from heapq import heappush, heappop

    smi = smi.copy()
    fmi = fmi.copy()

    n_it = np.ceil(n/2)
    interactions = []

    # Use a max-heap to store the indices and values for SMI and FMI
    # Heaps in Python are min-heaps, so we use negative values for max-heap behavior
    smi_heap = []
    fmi_heap = []
    
    # Push the SMI values and their indices to a heap (for SMI)
    for i in range(smi.shape[0]):
        for j in range(smi.shape[1]):
            heappush(smi_heap, (-smi[i, j], i, j))  # Store negative for max heap
            heappush(fmi_heap, (-fmi[i, j], i, j))  # Store negative for max heap
    
    # Extract interactions efficiently using the heaps
    for i in range(int(n_it)):
        if smi_heap:
            smi_value, smi_i, smi_j = heappop(smi_heap)
            element1 = info["names"][smi_i]
            element2 = info["names"][smi_j]
            type1 = info["types"][smi_i]
            type2 = info["types"][smi_j]
            interactions.append({
                "Node1": element1,
                "Node2": element2,
                "SMI": -smi_value,  # Revert the negation
                "FMI": fmi[smi_i, smi_j],  # Original FMI value
                "Type1": type1,
                "Type2": type2
            })
            smi[smi_i, smi_j] = 0  # Set to zero to avoid re-selection
        
        if fmi_heap:
            fmi_value, fmi_i, fmi_j = heappop(fmi_heap)
            element1 = info["names"][fmi_i]
            element2 = info["names"][fmi_j]
            type1 = info["types"][fmi_i]
            type2 = info["types"][fmi_j]
            interactions.append({
                "Node1": element1,
                "Node2": element2,
                "SMI": smi[fmi_i, fmi_j],  # Original SMI value
                "FMI": -fmi_value,  # Revert the negation
                "Type1": type1,
                "Type2": type2
            })
            fmi[fmi_i, fmi_j] = 0  # Set to zero to avoid re-selection
    
    interactions_df = pd.DataFrame(interactions)
    return interactions_df
