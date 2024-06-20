import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.stats import norm
from sklearn import metrics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from numpy import inf
from networkx.drawing.nx_pydot import graphviz_layout
from auxiliary_functions import ens_to_gen
from gtfparse import read_gtf
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.patches as mpatches
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def order_interactions(s_mat, f_mat):
    
    n_nodos = s_mat.shape[0]

    mats = [s_mat, f_mat]

    connects_s = []
    connects_f = []

    ########################
    ##   ORDENAR LISTAS   ##
    ########################

    loop = 0
    # Para estructural y funcional
    for mat in mats:

        nodos = np.nonzero(mat)
        row_counts = np.count_nonzero(s_mat, axis=1)
        col_counts = np.count_nonzero(s_mat, axis=0)

        n = len(nodos[0])

        for i in range(n):
            nodo0 = nodos[0][i]
            nodo1 = nodos[1][i]
            vecinos = row_counts[nodo0] + col_counts[nodo1]
            if loop == 0:
                connects_s.append([nodo0, nodo1, mat[nodo0, nodo1], vecinos])
            else:
                connects_f.append([nodo0, nodo1, mat[nodo0, nodo1], vecinos])

        loop = loop + 1

    # Estructural de forma descendente según IM y ascendente según nº de vecinos.
    connects_s.sort(key=lambda x: (-x[2], x[3]))
    # Funcional de forma descendente según IM.
    connects_f.sort(key=lambda x: -x[2])

    combinado = [item for pair in zip(connects_s, connects_f) for item in pair]
    unique = []

    combi = [combined[0:2] for combined in combinado]

    for con in combi:
        if con not in unique:
            unique.append([con[0], con[1], s_mat[con[0], con[1]], f_mat[con[0], con[1]]])

    return unique


def structure_search(dataset_train, max_models, conexiones, dataset_test = None, link_txt = None):

    if dataset_test == None:

        accuracy_train = []
        ll = []
        bic = []
        auc = []

        print('GENERANDO DAGS...')
        print('NO TEST')

        for n_dag in tqdm(range(1,max_models+1)):

            n_nodos = len(dataset_train['data'].columns)
            ndata = len(dataset_train['data'])
            classvalues = dataset_train['classvalues']

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

            dataset_tr = dataset_train['data'].iloc[:, nodos_dag]

            # sigmas, betas = structure_search.clg_param_continuous(classes[:,0], dataset_tr, dag_bin)
            # sigmas2, betas2 = structure_search.clg_param_continuous(classes[:,1], dataset_tr, dag_bin)

            clgc = pl_plemclg(dataset_tr, classes, dag_bin)
            rl_train = pl_clgclassify(dataset_tr,params=clgc)
            accuracy_train.append(accuracy_score(dataset_train['classvalues'], rl_train['classification']))
            ll.append(clgc['ll'])
            bic.append(clgc['bic'])
            roc_auc = roc_auc_score(classvalues, rl_train['posterior'][:, 1])
            auc.append(roc_auc)

            if link_txt != False:

                with open(link_txt, 'a') as f:
                    sys.stdout = f
                    print('\nNº NODES DAG: ', n_dag)
                    print('Accuracy: ', accuracy_score(dataset_train['classvalues'], rl_train['classification']), 'Log Likelihood: ', clgc['ll'], 'BIC: ', clgc['bic'], 'AUC', roc_auc)
                    sys.stdout = sys.__stdout__

        if link_txt != False:
                
            max_acc = max(accuracy_train)
            max_acc_index = accuracy_train.index(max_acc)
            max_auc = max(auc)
            max_auc_index = auc.index(max_auc)

            with open(link_txt, 'a') as f:
                sys.stdout = f
                print('Max. Accuracy: ', max_acc, 'Nodes DAG: ', max_acc_index)
                print('Max. AUC: ', max_auc, 'Nodes DAG: ', max_auc_index)
                sys.stdout = sys.__stdout__

        return accuracy_train, ll, bic, auc

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

        print('GENERANDO DAGS...')
        print('SI TEST')

        for n_dag in tqdm(range(1,max_models+1)):

            n_nodos = len(dataset_train['data'].columns)
            ndata = len(dataset_train['data'])
            ndata_test = len(dataset_test['data'])
            classvalues = dataset_train['classvalues']
            classvalues_test = dataset_test['classvalues']

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

            dataset_tr = dataset_train['data'].iloc[:, nodos_dag]
            dataset_te = dataset_test['data'].iloc[:, nodos_dag]

            # sigmas, betas = structure_search.clg_param_continuous(classes[:,0], dataset_tr, dag_bin)
            # sigmas2, betas2 = structure_search.clg_param_continuous(classes[:,1], dataset_tr, dag_bin)

            acc_rf, acc_svc, acc_knn, acc_gb = compute_models(dataset_tr, dataset_te, dataset_train['classvalues'], dataset_test['classvalues'])
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
            accuracy_train.append(accuracy_score(dataset_train['classvalues'], rl_train['classification']))
            accuracy_test.append(accuracy_score(dataset_test['classvalues'], rl_test['classification']))
            posterior_prob_test = rl_test['posterior']
            prob_ll = classes_test * posterior_prob_test
            sumprob = np.sum(prob_ll, axis=1)
            ll = np.sum(np.log(sumprob))
            bic = ll-(np.log(dataset_te.shape[1])/2)*dimension(dag_bin,k)
            roc_auc = roc_auc_score(dataset_test['classvalues'], rl_test['posterior'][:, 1])
            auc_test.append(roc_auc)
            ll_test.append(ll)
            bic_test.append(bic)

            if link_txt != False:

                with open(link_txt, 'a') as f:
                    sys.stdout = f
                    print('\nNº NODES DAG: ', n_dag)
                    print('Accuracy train: ', accuracy_score(dataset_train['classvalues'], rl_train['classification']), 'Accuracy test: ', accuracy_score(dataset_test['classvalues'], rl_test['classification']))
                    sys.stdout = sys.__stdout__

        if link_txt != False:
                
            max_acc_train = max(accuracy_train)
            max_acctr_index = accuracy_train.index(max_acc_train)
            max_acc_test = max(accuracy_test)
            max_accte_index = accuracy_test.index(max_acc_test)

            with open(link_txt, 'a') as f:
                sys.stdout = f
                print('Max. Accuracy Train: ', max_acc_train, 'Nodes DAG: ', max_acctr_index)
                print('Max. Accuracy Test: ', max_acc_test, 'Nodes DAG: ', max_accte_index)
                sys.stdout = sys.__stdout__
    
        return accuracy_train, accuracy_test, accuracy_rf, accuracy_svc, accuracy_knn, accuracy_gb, auc_test, ll_test, bic_test



"""
FUNCIONALIDAD

La función structure_search() ejecuta la función pl_plemclg() para cada red a crear. La función pl_plemclg() 
redirige a las funciones clg_param_continuous() y pl_factorizedpdf() para calcula los parámetros del modelo
y las probabilidades de cada individuo. Con la información que recibe calcula las métricas del DAG (acc, ll, auc)
y se las devuelve a structure_search().


INPUT

dataset: Matriz con los datos de expresión de los nodos a incluir en el DAG.
classpl: Matriz con k columnas (tantas como clases) en la que cada elemento en la columna será un 1 si el elemento pertenece a dicha clase y 0 si no.
dag: Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la hay.


OUTPUT

rl: Diccionario con los parámetros del modelo y otra información.
    1) 'classp': Probabilidad a priori de cada clase.
    2) 'beta': Matriz nxn (siendo n el número de nodos del DAG) con los betas del modelo.
    3) 'sigma': Array con los sigmas del modelo.
    4) 'dag': Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la hay.
    5) 'll': Log-verosimilitud del DAG.
    6) 'acc': Accuracy del DAG.

"""

def pl_plemclg(dataset, classpl, dag):


    m = dataset.shape[0]
    n = dataset.shape[1]
    k = classpl.shape[1]

    classp = np.zeros(k)
    sigma = np.zeros((n,k))
    beta = np.zeros((n,n,k))

    pdf = np.zeros((m,k))

    for c in range(0,k):
            
        # Probabilidad a priori
        classp[c] = np.sum(classpl[:,c], axis=0)
        classp[c] /= m

        # Estimación de parámetros
        # print(dataset)
        variance, betas = clg_param_continuous(classpl[:,c], dataset, dag)

        sigma[:, c] = np.transpose(variance).ravel()
        
        beta[:, :, c] = betas

        pdf[:,c] = pl_factorizedpdf(dataset,beta[:,:,c], sigma[:,c], dag)

    replicated_classp = np.tile(classp, (m, 1))
    # classpl es un vector con k columnas que en cada columna muestra un 1 si ese elemento pertenece a esa clase y 0 si no.
    # replicated_classp repite la probabilidad a priori de cada clase m veces
    # pdf es la probabilidad asociada a cada elemento y cada clase
    # Se multiplica por la probabilidad a priori (sólo la probabilidad de su clase, la otra se anula al *0)
    posterior_prob = replicated_classp * pdf

    maxclass = np.argmax(posterior_prob, axis=1)
    # Se coje la segunda columna porque se le ha dado la vuelta al vector binario para que los 1 correspondan con cada clase, por tanto las clases originales se encuentran en la columna 1.
    accuracy = np.sum(classpl[:,1] == maxclass)/len(classpl[:,1])
    # fpr, tpr, thresholds = metrics.roc_curve(classpl[:,1], maxclass)
    # roc_auc = metrics.auc(fpr, tpr)

    # Se queda con la probabilidad de la clase real.
    prob_ll = classpl * posterior_prob
    sumprob = np.sum(prob_ll, axis=1)
    ll = np.sum(np.log(sumprob))
    bic = ll-(np.log(n)/2)*dimension(dag,k)

    rl = {"classp":classp,"beta":beta,"sigma":sigma,"dag":dag,"ll":ll,"acc":accuracy,"bic":bic}

    return rl


"""
FUNCIONALIDAD

clg_param_continuous() induce los parámetros de un modelo CLG.


INPUT

classes: Array binario (0/1) que indica si un elemento pertenece a la clase (1) o no (0). La función sólamente tiene en cuenta a aquellos elementos que sí pertenecen.
dataset: Matriz con los valores de expresión de los individuos en los microRNA y genes que están presentes en el DAG.
dag: Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la ha

OUTPUT

variance: Array con las varianzas de los nodos.
betamat: Matriz con los valores de los betas del modelo.

"""

def clg_param_continuous(classes, dataset, dag):

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
                            
                # PARCHE PARA MATRICES SINGULARES
                            
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


"""
FUNCIONALIDAD

pl_factorizedpdf() calcula la probabilidad asociada a cada individuo según un DAG dado.


INPUT

data: Matriz con los datos de expresión de los n nodos del DAG.
beta: Betas del modelo para una clase dada. Matriz n x n.
sigma: Sigmas del modelo para una clase dada. Array n x 1.
dag: Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la ha

OUTPUT

pdf: Devuelve la probabilidad asociada a cada individuo según el DAG.

"""

def pl_factorizedpdf(data, beta, sigma, dag):

    # print('ENTRA FACTORIZED')
    # Nº de individuos
    m = data.shape[0]
    # Nº de nodos
    n = dag.shape[1]
    pdf = np.ones(m)
    # Se convierten las varianzas en desviaciones
    stdevs = np.sqrt(sigma)
    # Para cada nodo del DAG
    for i in range(n):
        # Se buscan sus padres
        pa = np.where(dag[:,i] == 1)[0]
        # No tiene padres :(
        if len(pa) == 0:
            # La probabilidad depende sólamente de los valores del propio nodo (su media y desv) porque no tiene padres.
            # Se identifica la probabilidad de cada expresión (nº individuos) en base a su distribución (1 x nindividuos).
            pdfaux = norm.pdf(data.iloc[:,i], loc=beta[i][i], scale=stdevs[i])
        # Si tiene padres :)
        else:
            # Comprobamos cuantos padres tiene
            npa = len(pa)
            # Repetimos m veces su desviación (1 para cada individuo)
            stdev = np.repeat(stdevs[i], m)
            # Repetimos m veces el Beta0 del elemento
            mu = np.repeat(beta[i][i], m)
            # Unimos los índices de los padres con la posición del nodo correspondiente repetido npa veces (para obtener los índices con los que buscar en la matriz de parámetros)
            replicated_i = np.repeat(i, npa)
            ind = np.column_stack((replicated_i, pa))
            # Se obtienen con la matriz de betas los parámetros de las distribuciones de los padres (los betas)
            result = beta[ind[:, 0], ind[:, 1]]
            # Y se repite tantas veces como datos hay
            replicated_beta = np.tile(result, (m, 1))
            # Se multiplican los betas por los valores de expresión de los datos
            product_matrix = replicated_beta * data.iloc[:, pa]
            sum_vector = np.sum(product_matrix, axis=1)
            # Se le suma el beta0
            mu += sum_vector
            # Y se obtiene la probabilidad asociada en la distribución
            pdfaux = norm.pdf(data.iloc[:,i], loc=mu, scale=stdev)

        pdf = pdf*pdfaux

    return pdf


"""
FUNCIONALIDAD

pl_clgclassify() calcula las probabilidades de los individuos para cada clase y las une en una misma matriz.
Luego, llama a la función pl_classifyindep().

Esta función creo que estaría de más y se podría unificar de alguna forma.


INPUT

data: Matriz con los valores de expresión de los individuos en los microRNA y genes que están presentes en el DAG.
params: Un diccionario con los parámetros del modelo. Es la salida de pl_plemclg().


OUTPUT

rl: Diccionario con dos 'keys':
    1) 'posterior': La probabilidad posterior de cada clase para cada individuo.
    2) 'maxclass': Clase de mayor probabilidad de cada individuo y, por tanto, su predicción.
    Es la salida de pl_classifyindep().

"""

def pl_clgclassify(data,params):
    
    # print('ENTRA CLASSIFY')

    classp = params["classp"]
    beta = params["beta"]
    sigma = params["sigma"]
    dag = params["dag"]
    
    # Nº de datos
    m = data.shape[0]
    # Nº de nodos
    n =  data.shape[1]
    # Nº de clases
    k = len(classp)

    # if len(dag) == 1 ?¿

    condpdfx = np.zeros((m,k))

    for c in range(0,k):
        condpdfx[:,c] = pl_factorizedpdf(data, beta[:,:,c], sigma[:,c], dag)

    rl = pl_classifyindep(condpdfx, params)

    return rl




def pl_classifyindep(condpdfx, params):


    """
    Dadas las probabilidades de los individuos para cada clase y la probabilidad a priori de cada clase, pl_classifyindep()
    calcula la probabilidad a posteriori de cada individuo y realiza la predicción (clase más probable).

    Args:

        condpdfx (numpy.ndarray): Probabilidad asociada a cada clase para cada individuo. Matriz m x k (siendo m el número de individuos y k el de clases).
        params: (dict): Un diccionario con los parámetros del modelo.

                1) 'classp': Probabilidad a priori de cada clase.
                2) 'beta': Matriz nxn (siendo n el número de nodos del DAG) con los betas del modelo.
                3) 'sigma': Array con los sigmas del modelo.
                4) 'dag': Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la hay.
                5) 'll': Log-verosimilitud del DAG.
                6) 'acc': Accuracy del DAG. Las clases son Es la salida de pl_plemclg() (sólo se utiliza la probabilidad a priori).

                Es la salida de pl_plemclg() (sólo se utiliza la probabilidad a priori, 'classp').

    Returns:

        rl (dict): Diccionario con dos 'keys':

                1) 'posterior': La probabilidad posterior de cada clase para cada individuo.
                2) 'maxclass': Clase de mayor probabilidad de cada individuo y, por tanto, su predicción.

    """


    # Cargamos la probabilidad a priori de cada clase.
    classp = params["classp"]

    # Número de instancias
    m = condpdfx.shape[0]
    # Número de clases
    k = len(classp)

    # Se obtiene la probabilidad posterior
    posterior = np.zeros((m,k))
    for c in range(0,k):
        # Probabilidad a priori * Probabilidad modelo
        posterior[:,c] = classp[c]*condpdfx[:,c]

    # Obtenemos m probabilidades a posteriori para cada clase, una por cada individuo. Es la probabilidad asociada a cada clase para cada individuo.
    
    # Tsum incluye la suma de las probabilidades asociadas a cada clase
    tsum = np.sum(posterior,axis=1)
    # Sólo se queda con las sumas positivas (puede haber alguna negativa?¿?¿?¿)
    selec = tsum > 0

    # Relativiza los valores de cada clase
    posterior[selec, :] /= tsum[selec, np.newaxis]

    # Iguala a 0 las probabilidades negativas
    posterior[~selec, :] = 0
    # En los casos en los que había una probabilidad negativa (que ha sido asignada a 0), se asigna la otra a 1.
    posterior[~selec, np.argmax(classp)] = 1
    # Se busca la columna con la mayor probabilidad y se utiliza esa como la clase que se predice para la instancia.
    maxclass = np.argmax(posterior, axis=1)

    rl = {"posterior":posterior,"classification":maxclass}

    return rl



def dimension(dag, k):

    """
    Calcula la dimensión del DAG según el número de nodos y el número de clases. Se utiliza especificamente para la corrección del BIC.

    Args:

        dag (numpy.ndarray): Matriz binaria n x n, siendo n el número de nodos del DAG. El elemento ij será 1 si existe relación entre los nodos i y j y 0 si no la hay.
        k (int): Número de clases (fenotipos).

    Returns:

        dim (int): Dimensión del modelo. 

    """

    n = dag.shape[1]
    dim = k-1

    for i in range(n):
        npa = np.sum(dag[:, i])
        dim += npa * k + 2 * k 

    return dim



def fit_model(n_dag, dataset_train, conexiones):

    n_nodos = len(dataset_train['data'].columns)
    ndata = len(dataset_train['data'])
    classvalues = dataset_train['classvalues']

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
    dataset_tr = dataset_train['data'].iloc[:, nodos_dag]
    clgc = pl_plemclg(dataset_tr, classes, dag_bin)

    return nodos_dag, clgc

############

def predict_test(dataset_test, nodos_dag, clgc):

    """
    Realiza la predicción de un conjunto de test con los parámetros de un modelo CLG.

    Args:

        dataset_test (): 
        nodos_dag ():
        clgc ():

    Returns:

        rl (dict): Diccionario con dos 'keys':

            1) 'posterior': La probabilidad posterior de cada clase para cada individuo.
            2) 'maxclass': Clase de mayor probabilidad de cada individuo y, por tanto, su predicción.    

    """
     
    dataset_te = dataset_test['data'].iloc[:, nodos_dag]
    rl = pl_clgclassify(dataset_te,params=clgc)
    return rl

############

def compute_models(X_train, X_test, y_train, y_test):

    """
    Calcula la precisión de los modelos Random Forest, SVC, KNN y Gradient Boosting sobre unos conjuntos X e y.

    Args:

        X_train ():
        X_test ():
        y_train ():
        y_test():

    Returns:

        accuracy_rf ():
        accuracy_svc ():
        accuracy_knn ():
        accuracy_gb ():    

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




def show_connections(names, connections, k):

    """
    Representa la red de coregulación con k interacciones asociada a un modelo CLG.

    Args:

        names ():
        connections (int):
        k (int): Número de interacciones de la red.

    Returns:

         

    """
    
    from matplotlib.patches import Ellipse

    gtf = read_gtf("gtf/gencode.v45.chr_patch_hapl_scaff.basic.annotation.gtf")

    graph = nx.DiGraph()
    table = []
    # con_to_exp = pd.DataFrame(columns=['microRNA', 'gene', 'SMI', 'FMI'])
    # con_to_exp['microRNA'] = [names[connections[i][0]] for i in range(100)]
    # print('CONVERTING GENES...')
    # con_to_exp['gene'] = [ens_to_gen(names[connections[i][1]], gtf) for i in tqdm(range(100))]
    # con_to_exp['SMI'] = [round(connections[i][2],4) for i in range(100)]
    # con_to_exp['FMI'] = [round(connections[i][3],4) for i in range(100)]

    print('PLOTING GRAPH...')
    for i in tqdm(range(k)):

        gen = ens_to_gen(names[connections[i][1]], gtf)

        table.append([names[connections[i][0]], gen, round(connections[i][2],4), round(connections[i][3],4)])
        # table.append([names[connections[i][0]], names[connections[i][1]], round(connections[i][2],4), round(connections[i][3],4)])
        
        if names[connections[i][0]] not in graph.nodes:
            graph.add_node(names[connections[i][0]])

        if names[connections[i][1]] not in graph.nodes:
            # graph.add_node(names[connections[i][1]])
            graph.add_node(gen)
        
        # graph.add_edge(names[connections[i][0]], names[connections[i][1]], weight=round(connections[i][2],4))
        graph.add_edge(names[connections[i][0]], gen, value1=f'S {round(connections[i][2],3)}', value2=f'F {round(connections[i][3],3)}', linestyle='--')
    

    pos = graphviz_layout(graph, prog='neato')

    # nx.draw(graph, pos, with_labels=False, node_color='blue', edge_color='gray', node_size=800, font_size=16)

    plt.figure(figsize=(12, 9))

    # Dibujar las aristas
    nx.draw_networkx_edges(graph, pos, width=0.5)

    ax = plt.gca()
    
    # Dibujar los nodos como elipses y añadir etiquetas
    for node, (x, y) in pos.items():
        if node.startswith('hsa'):
            ellipse = Ellipse((x, y), width=45, height=40, edgecolor='white', facecolor='lightgreen', zorder=2, linewidth=1.5)
        else:
            ellipse = Ellipse((x, y), width=45, height=40, edgecolor='white', facecolor='lightblue', zorder=2, linewidth=0.5)

        ax.add_patch(ellipse)
        if node.startswith('hsa'):
            ax.text(x, y, node[4:], horizontalalignment='center', verticalalignment='center', zorder=3, fontsize=6)
        else:
            ax.text(x, y, node, horizontalalignment='center', verticalalignment='center', zorder=3, fontsize=6)
    
    for (u, v, d) in graph.edges(data=True):
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        annotation_text1 = f"{d['value1']}"
        annotation_text2 = f"{d['value2']}"
        # plt.annotate(annotation_text1, (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        # plt.annotate(annotation_text2, (x, y), textcoords="offset points", xytext=(0,-5), ha='center', fontsize=6, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        plt.annotate(annotation_text1, (x, y), textcoords="offset points", xytext=(0,3), ha='center', fontsize=6)
        plt.annotate(annotation_text2, (x, y), textcoords="offset points", xytext=(0,-3), ha='center', fontsize=6)

    green_patch = mpatches.Patch(color='lightgreen', label='MicroRNA')
    blue_patch = mpatches.Patch(color='lightblue', label='Gene')

    plt.legend(handles=[green_patch, blue_patch], frameon='False')
    
    plt.axis('off')  # Ocultar ejes
    plt.savefig('Results/Otros/network_prueba.svg', format='svg')

    plt.show()