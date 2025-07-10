import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.stats import norm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from networkx.drawing.nx_agraph import graphviz_layout




def structure_search(X_train, y_train, X_test = None, y_test = None, max_models = None, conexiones = None):

    """
    Checks networks with different number of interactions and return their metrics.

    Args:

            X_train (pd.DataFrame): Dataframe with the train expression values.
            y_train (pd.Series): Series with the class of each individual in the train set.
            X_test (pd.DataFrame): Dataframe with the test expression values.
            y_test (pd.Series): Series with the class of each individual in the test set.
            max_models (int): Maximum number of interactions to check.
            conexiones (list): List with the interactions ordered by importance.
    
    Returns:

            result (pd.DataFrame): Dataframe with the metrics of the networks generated. If test set is specified metrics of alternative models are included.
    
    """

    if not isinstance(X_test, pd.DataFrame) or not isinstance(y_test, pd.Series):

        accuracy_train = []
        ll = []
        bic = []
        auc = []
        sensi_tr = []
        speci_tr = []

        # print('GENERANDO DAGS...')
        # print('NO TEST')

        for n_dag in tqdm(range(1,max_models+1)):

            n_nodos = len(X_train.columns)
            ndata = len(X_train)
            classvalues = y_train

            k = np.max(classvalues) + 1
            k = int(k)

            classes = np.zeros((ndata, k))

            for i in range(0, ndata):
                for j in range(0,k):
                    # if classvalues.iloc[i,0] == j:
                    if classvalues.iloc[i] == j:
                        classes[i][j] = 1

            nodos_dag = []
            for conec in conexiones[0:n_dag]:
                if conec[0] not in nodos_dag:
                    nodos_dag.append(conec[0])
                if conec[1] not in nodos_dag:
                    nodos_dag.append(conec[1])

            dag_bin_general = np.zeros((n_nodos, n_nodos))

            for con in conexiones[0:n_dag]:
                dag_bin_general[con[0]][con[1]] = 1

            nodos_dag.sort()

            dag_bin = dag_bin_general[nodos_dag][:, nodos_dag]

            dataset_tr = X_train.iloc[:, nodos_dag]

            clgc = pl_plemclg(dataset_tr, classes, dag_bin)
            rl_train = pl_clgclassify(dataset_tr,params=clgc)
            accuracy_train.append(accuracy_score(y_train, rl_train['classification']))
            tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, rl_train['classification']).ravel()
            sensitivity_tr = tp_tr / (tp_tr + fn_tr)
            specificity_tr = tn_tr / (tn_tr + fp_tr)
            sensi_tr.append(sensitivity_tr)
            speci_tr.append(specificity_tr)
            ll.append(clgc['ll'])
            bic.append(clgc['bic'])
            roc_auc = roc_auc_score(classvalues, rl_train['posterior'][:, 1])
            auc.append(roc_auc)

        result = pd.DataFrame(list(zip(accuracy_train, ll, bic, auc, sensi_tr, speci_tr)), columns = ['Accuracy', 'Log Likelihood', 'BIC', 'AUC', 'Sensitivity', 'Specificity']) 

        return result

    else:

        accuracy_train = []
        accuracy_test = []
        accuracy_rf = []
        accuracy_svc = []
        accuracy_knn = []
        accuracy_gb = []
        ll_test = []
        bic_test = []
        auc_test = []
        sensi_test = []
        speci_test = []

        # print('GENERANDO DAGS...')
        # print('SI TEST')

        for n_dag in tqdm(range(1,max_models+1)):

            n_nodos = len(X_train.columns)
            ndata = len(X_train)
            ndata_test = len(X_test)
            classvalues = y_train
            classvalues_test = y_test

            k = np.max(classvalues) + 1
            k = int(k)

            classes = np.zeros((ndata, k))

            for i in range(0, ndata):
                for j in range(0,k):
                    # if classvalues.iloc[i,0] == j:
                    if classvalues.iloc[i] == j:
                        classes[i][j] = 1

            nodos_dag = []
            for conec in conexiones[0:n_dag]:
                if conec[0] not in nodos_dag:
                    nodos_dag.append(conec[0])
                if conec[1] not in nodos_dag:
                    nodos_dag.append(conec[1])

            dag_bin_general = np.zeros((n_nodos, n_nodos))

            for con in conexiones[0:n_dag]:
                dag_bin_general[con[0]][con[1]] = 1

            nodos_dag.sort()

            dag_bin = dag_bin_general[nodos_dag][:, nodos_dag]

            dataset_tr = X_train.iloc[:, nodos_dag]
            dataset_te = X_test.iloc[:, nodos_dag]

            acc_rf, acc_svc, acc_knn, acc_gb = compute_models(dataset_tr, dataset_te, y_train, y_test)
            accuracy_rf.append(acc_rf)
            accuracy_svc.append(acc_svc)
            accuracy_knn.append(acc_knn)
            accuracy_gb.append(acc_gb)

            classes_test = np.zeros((ndata_test, k))
            for i in range(0, ndata_test):
                for j in range(0,k):
                    # if classvalues.iloc[i,0] == j:
                    if classvalues_test.iloc[i] == j:
                        classes_test[i][j] = 1

            clgc = pl_plemclg(dataset_tr, classes, dag_bin)
            rl_train = pl_clgclassify(dataset_tr,params=clgc)
            rl_test = pl_clgclassify(dataset_te,params=clgc)
            accuracy_train.append(accuracy_score(y_train, rl_train['classification']))
            accuracy_test.append(accuracy_score(y_test, rl_test['classification']))
            tn, fp, fn, tp = confusion_matrix(y_test, rl_test['classification']).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            sensi_test.append(sensitivity)
            speci_test.append(specificity)
            posterior_prob_test = rl_test['posterior']
            prob_ll = classes_test * posterior_prob_test
            sumprob = np.sum(prob_ll, axis=1)
            ll = np.sum(np.log(sumprob))
            bic = ll-(np.log(dataset_te.shape[1])/2)*dimension(dag_bin,k)
            roc_auc = roc_auc_score(y_test, rl_test['posterior'][:, 1])
            auc_test.append(roc_auc)
            ll_test.append(ll)
            bic_test.append(bic)

        result = pd.DataFrame(list(zip(accuracy_train, accuracy_test, accuracy_rf, accuracy_svc, accuracy_knn, accuracy_gb, ll_test, bic_test, auc_test, sensi_test, speci_test)), columns = ['Accuracy Train', 'Accuracy Test', 'Accuracy RF', 'Accuracy SVC', 'Accuracy KNN', 'Accuracy GB', 'Log Likelihood', 'BIC', 'AUC', 'Sensitivity', 'Specificity'])

        return result




def order_interactions(s_mat, f_mat, mode='alternating', weight=0.5):

    """

    Orders the interactions of the network based on structural and/or functional importance.

    Args:
        s_mat (np.ndarray): Structural information matrix.
        f_mat (np.ndarray): Functional information matrix.
        mode (str): 'structural', 'functional', 'weighted', or 'alternating'.
        weight (float): Weight for structural score in 'combined' mode. Value between 0 and 1.

    Returns:
        unique (list): List of interactions [node1, node2, structural_score, functional_score] ordered by importance.

    """

    connects_s = []
    connects_f = []
    n = s_mat.shape[0]

    row_counts = np.count_nonzero(s_mat, axis=1)
    col_counts = np.count_nonzero(s_mat, axis=0)

    for i in range(n):
        for j in range(n):
            struct_score = s_mat[i, j]
            func_score = f_mat[i, j]

            if struct_score != 0 or func_score != 0:
                vecinos = row_counts[i] + col_counts[i] + row_counts[j] + col_counts[j]
                connects_s.append([i, j, struct_score, func_score, vecinos])
                connects_f.append([i, j, struct_score, func_score, vecinos])

    if mode == 'structural':
        # Sort by structural score (desc), then vecinos (asc)
        connects_s.sort(key=lambda x: (-x[2], x[4]))
        sorted_items = connects_s

    elif mode == 'functional':
        # Sort by functional score (desc)
        connects_f.sort(key=lambda x: -x[3])
        sorted_items = connects_f

    elif mode == 'weighted':
        combined = []
        for i, j, s, f, v in connects_s:
            combined_score = weight * s + (1 - weight) * f
            combined.append([i, j, s, f, combined_score, v])
        combined.sort(key=lambda x: (-x[4], x[5]))  # sort by combined score, then vecinos
        sorted_items = [[i, j, s, f] for i, j, s, f, _, _ in combined]

    elif mode == 'alternating':
        connects_s.sort(key=lambda x: (-x[2], x[4]))
        connects_f.sort(key=lambda x: -x[3])
        interleaved = [item for pair in zip(connects_s, connects_f) for item in pair]
        sorted_items = []
        seen = set()
        for i, j, s, f, _ in interleaved:
            key = tuple(sorted((i, j)))
            if key not in seen:
                sorted_items.append([i, j, s, f])
                seen.add(key)
        return sorted_items

    else:
        raise ValueError("Invalid mode. Choose from 'structural', 'functional', 'weighted', 'alternating'.")

    # For structural/functional/combined modes: remove duplicates
    seen = set()
    unique = []
    for i, j, s, f, *_ in sorted_items:
        key = tuple(sorted((i, j)))
        if key not in seen:
            unique.append([i, j, s, f])
            seen.add(key)

    return unique



def pl_plemclg(dataset, classpl, dag):

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

    for c in range(0,k):
            
        # A priori probability.
        classp[c] = np.sum(classpl[:,c], axis=0)
        classp[c] /= m

        # Parameter estimation.
        variance, betas = clg_param_continuous(classpl[:,c], dataset, dag)

        sigma[:, c] = np.transpose(variance).ravel()
        
        beta[:, :, c] = betas

        pdf[:,c] = pl_factorizedpdf(dataset,beta[:,:,c], sigma[:,c], dag)

    replicated_classp = np.tile(classp, (m, 1))
    # classpl is a vector with k columns that in each column shows a 1 if that element belongs to that class and 0 if not.
    # replicated_classp repeats the prior probability of each class m times.
    # pdf is the probability associated with each element and each class.
    # It is multiplied by the prior probability (only the probability of its class, the other is canceled by *0)
    posterior_prob = replicated_classp * pdf

    maxclass = np.argmax(posterior_prob, axis=1)
    # The second column is taken because the binary vector has been reversed so that the 1s correspond to each class, therefore the original classes are in column 1.
    accuracy = np.sum(classpl[:,1] == maxclass)/len(classpl[:,1])

    # We take the probability of the real class.
    prob_ll = classpl * posterior_prob
    sumprob = np.sum(prob_ll, axis=1)
    ll = np.sum(np.log(sumprob))
    bic = ll-(np.log(n)/2)*dimension(dag,k)

    rl = {"classp":classp,"beta":beta,"sigma":sigma,"dag":dag,"ll":ll,"acc":accuracy,"bic":bic}

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

    m = dataset.shape[0]
    n = dataset.shape[1]
    N = classes.sum()

    mu = np.zeros((1,n)) 
    variance = np.zeros((1,n)) 
    diffvar = np.zeros((m,n)) 
    covmat = np.zeros((n,n))
    betamat = np.zeros((n,n))

    for i in range(0,n):
        mu[0,i] = np.sum(classes*dataset.iloc[:,i]) / N
        diffvar[:,i] = dataset.iloc[:,i] - mu[0][i]
        covmat[i,i] = np.sum(classes*diffvar[:,i]*diffvar[:,i]) / N 

    for i in range(0,n):
        for j in range(0,n):
            if i != j:
                covmat[i,j] = np.sum(classes*diffvar[:,i]*diffvar[:,j]) / N

    for i in range(0,n):

        pa = np.where(dag[:, i] == 1)[0]
        npa = len(pa)

        if npa == 0:
            betamat[i,i] = mu[0][i]
            variance[0][i] = covmat[i,i]

        else:

            for parentindex in range(0, npa):

                j = pa[parentindex]
                covipa = covmat[i,pa]
                covpai = covmat[pa,i]
                covii = covmat[i,i]
                covpapa = covmat[np.ix_(pa, pa)]

            if npa == 1:
                div = 1 / covpapa
                betamat[i,i] = mu[0][i] - covipa * div * mu[0][pa]
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


def pl_clgclassify(data,params):

    """

    Given some data and the parameters of a CLG model, pl_clgclassify() calculates the probability associated with each class for each individual and makes the prediction (most probable class).
    
    Args:

        data (np.ndarray): Matrix with the expression values of the individuals in the microRNA and genes that are present in the DAG.
        params (dict): A dictionary with the following model parameters.          
                    1) 'classp': A priori probability of each class.
                    2) 'beta': Matrix nxn (being n the number of nodes in the DAG) with the betas of the model.
                    3) 'sigma': Array with the sigmas of the model.
                    4) 'dag': Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
                    5) 'll': Log-likelihood of the DAG.
                    6) 'acc': Accuracy of the DAG.

    Returns:

        rl (dict): Dictionary with two 'keys':
                    1) 'posterior': The posterior probability of each class for each individual.
                    2) 'maxclass': Class of greatest probability of each individual and, therefore, its prediction.

    """

    classp = params["classp"]
    beta = params["beta"]
    sigma = params["sigma"]
    dag = params["dag"]
    
    # Nº of data
    m = data.shape[0]
    # Nº of nodes
    n =  data.shape[1]
    # Nº of classes
    k = len(classp)

    condpdfx = np.zeros((m,k))

    for c in range(0,k):
        condpdfx[:,c] = pl_factorizedpdf(data, beta[:,:,c], sigma[:,c], dag)

    rl = pl_classifyindep(condpdfx, params)

    return rl




def pl_classifyindep(condpdfx, params):


    """
    
    Given the probabilities of the individuals for each class and the prior probability of each class, pl_classifyindep()
    calculates the posterior probability of each individual and makes the prediction (most probable class).

    Args:

        condpdfx (np.ndarray): Probability associated to each class for each individual. Matrix m x k (being m the number of individuals and k the number of classes).
        params (dict): A dictionary with model parameters.        
                    1) 'classp': A priori probability of each class.
                    2) 'beta': Matrix nxn (being n the number of nodes in the DAG) with the betas of the model.
                    3) 'sigma': Array with the sigmas of the model.
                    4) 'dag': Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
                    5) 'll': Log-likelihood of the DAG.
                    6) 'acc': Accuracy of the DAG.

    Returns:

        rl (dict): Dictionary with two 'keys':
                    1) 'posterior': The posterior probability of each class for each individual.
                    2) 'maxclass': Class of greatest probability of each individual and, therefore, its prediction.

    """


    # Charge the prior probability of each class.
    classp = params["classp"]

    # Number of instances
    m = condpdfx.shape[0]
    # Number of classes
    k = len(classp)

    # Get the posterior probability
    posterior = np.zeros((m,k))
    for c in range(0,k):
        # Probabilidad a priori * Probabilidad modelo
        # Prior probability * Model probability
        posterior[:,c] = classp[c]*condpdfx[:,c]

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
    posterior[~selec, np.argmax(classp)] = 1

    # The column with the highest probability is searched for and that is used as the class that is predicted for the instance.
    maxclass = np.argmax(posterior, axis=1)

    rl = {"posterior":posterior,"classification":maxclass}

    return rl



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



def fit_model(X, y, n_con, conexiones):

    """

    Fit a model with k interactions to the data.

    Args:

        X (pd.DataFrame): Dataframe with the expression values of the individuals in the microRNA and genes that are present in the DAG.
        y (pd.Series): Series with the class of each individual.
        n_con (int): Number of interactions of the network.
        conexiones (list): List with the interactions ordered by importance.

    Returns:

        nodos_dag (list): List with the nodes of the DAG.
        clgc (dict): dictionary with the parameters of the model.
            1) 'classp': Prior probability of each class.
            2) 'beta': Matrix nxn (being n the number of nodes in the DAG) with the betas of the model.
            3) 'sigma': Array with the sigmas of the model.
            4) 'dag': Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
            5) 'll': Log-likelihood of the DAG.
            6) 'acc': DAG accuracy.

    """

    n_nodos = len(X.columns)
    ndata = len(X)
    classvalues = y

    k = np.max(classvalues) + 1
    k = int(k)

    classes = np.zeros((ndata, k))

    for i in range(0, ndata):
        for j in range(0,k):
            if classvalues.iloc[i] == j:
                classes[i][j] = 1

    nodos_dag = []
    for conec in conexiones[0:n_con]:
        if conec[0] not in nodos_dag:
            nodos_dag.append(conec[0])
        if conec[1] not in nodos_dag:
            nodos_dag.append(conec[1])

    dag_bin_general = np.zeros((n_nodos, n_nodos))

    for con in conexiones[0:n_con]:
        dag_bin_general[con[0]][con[1]] = 1

    nodos_dag.sort()
    dag_bin = dag_bin_general[nodos_dag][:, nodos_dag]
    dataset_tr = X.iloc[:, nodos_dag]
    clgc = pl_plemclg(dataset_tr, classes, dag_bin)
    clgc['G'] = get_graph(dag_bin, dataset_tr)
    clgc['X'] = dataset_tr
    
    return nodos_dag, clgc




def predict_test(X_test, nodos_dag, clgc):

    """
    Make predictions on a test set with the parameters of a CLG model.

    Args:

        X_test (pd.DataFrame): Dataframe with the expression values of the individuals in the microRNA and genes.
        nodos_dag (list): List with the nodes of the DAG.
        clgc (dict): Dictionary with the model parameters and other information (output of pl_plemclg).
            1) 'classp': Prior probability of each class.
            2) 'beta': Matrix nxn (being n the number of nodes in the DAG) with the betas of the model.
            3) 'sigma': Array with the sigmas of the model.
            4) 'dag': Binary matrix n x n, where n is the number of nodes in the DAG. The element ij will be 1 if there is a relationship between nodes i and j and 0 if not.
            5) 'll': Log-likelihood of the DAG.
            6) 'acc': DAG accuracy.

    Returns:

        rl (dict): Dictionary with two 'keys':
            1) 'posterior': The posterior probability of each class for each individual.
            2) 'maxclass': Biggest probability class of each individual and, therefore, its prediction.   

    """
     
    dataset_te = X_test.iloc[:, nodos_dag]
    rl = pl_clgclassify(dataset_te,params=clgc)
    return rl



def compute_models(X_train, X_test, y_train, y_test):

    """
    Calculates the precision of the Random Forest, SVC, KNN and Gradient Boosting models.

    Args:

        X_train (pd.DataFrame): Dataframe with the train expression values.
        X_test (pd.DataFrame): Dataframe with the test expression values.
        y_train (pd.Series): Series with the class of each individual in the train set.
        y_test (pd.Series): Series with the class of each individual in the test set.

    Returns:

        accuracy_rf (float): Accuracy value of the Random Forest model.
        accuracy_svc (float): Accuracy value of the SVM classifier model.
        accuracy_knn (float): Accuracy of the KNN model.
        accuracy_gb (float): Accuracy of the Gradient Boosting model.    

    """

    # RANDOM FOREST

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)

    # SVC

    svc_classifier = SVC(kernel='linear', random_state=42)
    svc_classifier.fit(X_train, y_train)
    y_pred = svc_classifier.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred)

    # KNN

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred)

    # GRADIENT BOOSTING

    gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
    gb_classifier.fit(X_train, y_train)
    y_pred = gb_classifier.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred)

    return accuracy_rf, accuracy_svc, accuracy_knn, accuracy_gb



def get_graph(adj_matrix, train):

    """
    Converts an adjacency matrix into a MultiDiGraph.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix representing the connections in the network.
        train (pd.DataFrame): Dataframe with the expression values of the individuals.

    Returns:
        G (nx.MultiDiGraph): A MultiDiGraph object representing the network.

    """

    G = nx.MultiDiGraph()
    num_nodes = adj_matrix.shape[0]
    
    for i in range(num_nodes):
        G.add_node(i, label=train.columns[i])

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    return G



def get_network(connections, k, gtf, display):

    """

    Generates a network graph from the connections dataframe and saves it.

    Args:

        connections (pd.DataFrame): Dataframe with the connections of the network.
        k (int): Number of connections to include in the network.
        gtf (pd.DataFrame): Dataframe with the gene annotation information.
        display (bool): If True, displays the network graph.

    Returns:
        G (nx.MultiDiGraph): A MultiDiGraph object representing the network.

    """

    G = nx.MultiDiGraph()

    cons_def = connections.copy()

    cons_def['FMI'] = cons_def['FMI'].apply(lambda x: max(x, 0)) / np.max(cons_def['FMI'])

    mapping_dict = dict(zip(gtf['gene_id'].str.split('.').str[0], gtf['gene_name']))

    cons_def.iloc[:k, 1] = cons_def.iloc[:k, 1].map(lambda x: mapping_dict.get(x, x))

    nodes = set(cons_def.iloc[:k, 0]).union(set(cons_def.iloc[:k, 1]))

    for node in nodes:
        
        if str(node).startswith('hsa'):

            color = "lightgreen"

        else:
            
            color = 'skyblue'

        G.add_node(node, color=color)

    # Add edges with labels and keys

    for i in range(0, k):
        src = cons_def.iloc[i, 0]
        tgt = cons_def.iloc[i, 1]
        G.add_edge(src, tgt)

    if display:

        plt.figure(figsize=(16, 16))
        nodes = G.nodes
        pos = graphviz_layout(G, prog='neato')
        node_colors = [G.nodes[node]["color"] for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
        plt.axis('off')
        plt.show()

    return G




def get_ranking(names, connections, gtf):

    
    """
    Formats the interaction ranking as dataframe with miRNA and mRNA name and SMI and FMI values.

    Args:

        names (list): List with the names of the nodes in the network.
        connections (list): List with the ordered connections of the network as handled internly by the package.
        gtf (str): miRNA and mRNA annotation file.

    Returns:

        cons (pd.DataFrame): Dataframe with the connections of the network with the names of the nodes and the SMI and FMI values.

    """


    def reemplazar_numeros_por_nombres(conects, nombres):
        nueva_lista_de_listas = [
            [nombres[sublista[0]], nombres[sublista[1]]] + sublista[2:]
            for sublista in conects
        ]
        return nueva_lista_de_listas
    
    nueva_lista = reemplazar_numeros_por_nombres(connections, names)

    gtf['gene_id_stripped'] = gtf['gene_id'].str.split('.').str[0]

    mapping_dict = dict(zip(gtf['gene_id_stripped'], gtf['gene_name']))

    nueva_lista = [
        [sublist[0]] + [mapping_dict.get(sublist[1], sublist[1])] + sublist[2:]
        for sublist in nueva_lista
    ]   

    cons = pd.DataFrame(nueva_lista, columns=['miRNA', 'mRNA', 'SMI', 'FMI'])

    return cons

