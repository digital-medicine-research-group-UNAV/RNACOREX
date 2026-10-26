import numpy as np
from scipy.linalg import inv
from scipy.stats import norm


def learn_LGBN(dataset, classpl, dag, fmi, smi, info):

    """
    
    The structure_search() function executes the pl_plemclg() function for each network to create. The pl_plemclg() function redirects to the clg_param_continuous() 
    and pdf_lgbn() functions to calculate the model parameters and the probabilities of each individual. With the information it receives, 
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
    covmat = np.zeros((n,n,k))

    pdf = np.zeros((m,k))

    # Para cada clase

    for c in range(0,k):
            
        # Calcular la probabilidad a priori.

        classp[c] = np.sum(classpl[:,c], axis=0)
        classp[c] /= m

        # Estimar los parámetros.

        variance, betas, cmat = clg_param_continuous(classpl[:,c], dataset, dag)

        sigma[:, c] = np.transpose(variance).ravel()
        
        beta[:, :, c] = betas

        covmat[:, :, c] = cmat

        # Obtener probabilidades de cada muestra para cada clase.

        pdf[:,c] = pdf_lgbn(dataset,beta[:,:,c], sigma[:,c], dag)

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

    trueclass = np.argmax(classpl, axis=1)

    accuracy = np.mean(maxclass == trueclass)

    # We take the probability of the real class.

    prob_ll = classpl * posterior_prob

    # Sum all probabilities for real classes.

    sumprob = np.sum(prob_ll, axis=1)

    ll = np.sum(np.log(sumprob))

    bic = ll-(np.log(n)/2)*dimension(dag,k)

    rl = {"classp":classp,"beta":beta,"sigma":sigma,"covmat":covmat,"dag":dag,"labels":info["labels"],"types":info["types"],"distribution":None,"names":info["names"],"fmi":fmi,"smi":smi,"ll":ll,"acc":accuracy,"bic":bic}

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

    return variance, betamat, covmat



def pdf_lgbn(data, beta, sigma, dag):

    
    """

    Given some data and the parameters of a CLG model, pdf_lgbn() calculates the probability associated with each individual.
    
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