from scipy.stats import norm, gaussian_kde
from scipy.integrate import nquad
from numpy import trapz
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import warnings


warnings.filterwarnings("ignore")

def individual_mutual_info_trapz(micro, gen, datos, precision=100, step = 0, bandwidth = 'scott', plot_x = False, plot_y = False, plot_xy = False):

    """
    The function calculates the mutual information between a microRNA and a gene. To do this, it estimates the distribution of each element using Gaussian kernels.
    
    Args:
    
        micro (int): Index of the microRNA to be analyzed.
        gen (int): Index of the gene to be analyzed.
        datos (pd.DataFrame): Dataframe with the expression of the microRNAs and genes.
        precision (int): Number of points at which the function is evaluated when calculating numerical integration.
        step (int): Length of the step between each evaluation point. IF STEP IS INDICATED, THE INTEGRATION IS ACCORDING TO THIS PARAMETER AND THE ESTABLISHED PRECISION IS NOT TAKEN INTO ACCOUNT.
        bandwidth (str): Method to define the bandwidth when adjusting the kernel ('scott', 'silverman' and others accepted by scipy.stats.gaussian_kde).
        plot_x (bool): If True is indicated, the marginal distribution estimated by kernels of the microRNA is plotted and saved.
        plot_y (bool): If True is indicated, the marginal distribution estimated by kernels of the gene is plotted and saved.
        plot_xy (bool): If True is indicated, the joint micro-gene distribution estimated by kernels is plotted and saved.
        
    Returns:
    
        im (float): Value of the mutual information between the indicated microRNA and gene.
    """

    dataset = {"data": datos.iloc[:, :-1], "classvalues": datos.iloc[:, -1:]}

    data = dataset['data']
    data['classvalues'] = dataset['classvalues']

    mirna = data.iloc[:,micro]
    target = data.iloc[:,gen]

    # A very small value is defined to avoid division by 0.
    MIN_DOUBLE = 4.9406564584124654e-324 

    # The proportion of elements per class.
    pz1 = dataset['classvalues'].sum() / len(dataset['classvalues'])
    pz0 = 1 - pz1

    gen1 = data[data['classvalues']==1].iloc[:,gen]
    mic1 = data[data['classvalues']==1].iloc[:,micro]
    gen0 = data[data['classvalues']==0].iloc[:,gen]
    mic0 = data[data['classvalues']==0].iloc[:,micro]

    f_xz0_kde = gaussian_kde(mic0, bw_method = bandwidth)
    f_yz0_kde = gaussian_kde(gen0, bw_method = bandwidth)
    f_xyz0_kde = gaussian_kde(np.vstack((mic0, gen0)), bw_method = bandwidth)
    f_xz1_kde = gaussian_kde(mic1, bw_method = bandwidth)
    f_yz1_kde = gaussian_kde(gen1, bw_method = bandwidth)
    f_xyz1_kde = gaussian_kde(np.vstack((mic1, gen1)), bw_method = bandwidth)

    # integrand0 = lambda a,b: f_xyz0_kde.evaluate([a,b]) * math.log((f_xyz0_kde.evaluate([a,b]) / ((f_xz0_kde.evaluate(a)*f_yz0_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)
    # integrand1 = lambda a,b: f_xyz1_kde.evaluate([a,b]) * math.log((f_xyz1_kde.evaluate([a,b]) / ((f_xz1_kde.evaluate(a)*f_yz1_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)

    def integrand0_v2(a,b):
        res0 = f_xyz0_kde.evaluate([a,b]) * math.log((f_xyz0_kde.evaluate([a,b]) / ((f_xz0_kde.evaluate(a)*f_yz0_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)
        return res0

    def integrand1_v2(a,b):
        res1 = f_xyz1_kde.evaluate([a,b]) * math.log((f_xyz1_kde.evaluate([a,b]) / ((f_xz1_kde.evaluate(a)*f_yz1_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)
        return res1

    if step == 0:
        x_pts = np.linspace(mirna.min()-mirna.std(), mirna.max()+mirna.std(), precision)
        y_pts = np.linspace(target.min()-target.std(), target.max()+target.std(), precision)
        s = (precision,precision)
        integrand0 = np.zeros(s)
        integrand1 = np.zeros(s)
    
    else:
        x_pts = np.arange(mirna.min()-mirna.std(), mirna.max()+mirna.std(), step)
        y_pts = np.arange(target.min()-target.std(), target.max()+target.std(), step)
        s = (len(x_pts),len(y_pts))
        integrand0 = np.zeros(s)
        integrand1 = np.zeros(s)

    i = -1
    for x in x_pts:
        i += 1
        j = 0
        for y in y_pts:
            integrand0[i,j] = integrand0_v2(x,y)[0]
            integrand1[i,j] = integrand1_v2(x,y)[0]
            j += 1
    
    if plot_x == True:
        f_x_kde = gaussian_kde(mirna)
        eval_points = np.linspace(mirna.min()-mirna.std(), mirna.max()+mirna.std())
        y_sp = f_x_kde.pdf(eval_points)
        plt.figure(figsize=(8, 6))
        plt.plot(eval_points, y_sp)
        plt.title('Probability density estimate for kernel X')
        plt.savefig('kernel_x.png')

    if plot_y == True:
        f_y_kde = gaussian_kde(target)
        eval_points = np.linspace(target.min()-target.std(), target.max()+target.std())
        y_sp = f_y_kde.pdf(eval_points)
        plt.figure(figsize=(8, 6))
        plt.plot(eval_points, y_sp)
        plt.title('Probability density estimate for kernel Y')
        plt.savefig('kernel_y.png')

    if plot_xy == True:

        f_x_kde = gaussian_kde(mirna)
        f_y_kde = gaussian_kde(target)
        f_xy_kde = gaussian_kde(np.vstack((mirna, target)))
        x_pts = np.linspace(mirna.min()-mirna.std(), mirna.max()+mirna.std(), precision)
        y_pts = np.linspace(target.min()-target.std(), target.max()+target.std(), precision)
        x, y = np.mgrid[mirna.min()-mirna.std():mirna.max()+mirna.std():precision*1j, target.min()-target.std():target.max()+target.std():precision*1j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(f_xy_kde(positions).T, x.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, z, cmap='Blues')
        plt.colorbar()
        plt.title('Probability density estimate for bidimensional kernel (2D KDE)')
        plt.savefig('kernel_xy.png')


    return (pz0 * np.trapz(np.trapz(integrand0, x_pts, axis=0), y_pts, axis=0) + pz1 * np.trapz(np.trapz(integrand1, x_pts, axis=0), y_pts, axis=0)).values[0]



def mutual_info_trapz_matrix_scikit(X_train, y_train, structural_information, precision = 100, save = 'URL'):

    """

    The function calculates the mutual information between those microRNA-gene pairs that have structural information. To do this, it estimates the distribution of each element using Gaussian kernels
    
    Args:
    
            X_train (pd.DataFrame): Dataframe with the expression of the microRNAs and genes.
            y_train (pd.Series): Dataframe with the classes/phenotypes.
            structural_information (np.ndarray): Matrix with the structural information.
            precision (int): Number of points at which the function is evaluated when calculating numerical integration.
            save (str): URL where the matrix will be saved.
    
    OUTPUTS:
    
            functional_information (np.ndarray): Matrix with the functional information between each pair of nodes. The matrix will be of dimension (n+m)*(n+m) being n = number of microRNA and m = number of genes.
    
    """

    data = X_train

    # A very small value is defined to avoid division by 0.

    MIN_DOUBLE = 4.9406564584124654e-324 

    nodos0 = np.where(structural_information != 0)[0]
    nodos1 = np.where(structural_information != 0)[1]

    # The proportion of elements per class.

    pz1 = y_train.sum() / len(y_train)
    pz0 = 1 - pz1

    # The micros and genes are divided by classes.

    datos0 = X_train[(y_train == 0).values]
    datos1 = X_train[(y_train == 1).values]

    # 
    n_nodos = X_train.shape[1]
    functional_information = np.zeros((n_nodos, n_nodos))

    # Arrays are defined to save the values.

    s = (precision,precision)
    integ0 = np.zeros(s)
    integ1 = np.zeros(s)

    print('\nCALCULATING FUNCTIONAL MUTUAL INFORMATION...')
    for i in tqdm(range(len(nodos0))):
        
        try:
            nodo0 = nodos0[i]
            nodo1 = nodos1[i]

            # Kernel associated with the microRNA of the connection (P(X|Z)).
            f_xz1_kde = gaussian_kde(datos1.iloc[:,nodo0])
            f_xz0_kde = gaussian_kde(datos0.iloc[:,nodo0])

            # Kernel associated with the gene of the connection (P(Y|Z)).
            f_yz1_kde = gaussian_kde(datos1.iloc[:,nodo1])
            f_yz0_kde = gaussian_kde(datos0.iloc[:,nodo1])

            # Bidimensional kernel with both elements (P(X,Y|Z)).
            f_xyz0_kde = gaussian_kde(np.vstack((datos0.iloc[:,nodo0], datos0.iloc[:,nodo1]))) 
            f_xyz1_kde = gaussian_kde(np.vstack((datos1.iloc[:,nodo0], datos1.iloc[:,nodo1]))) 

            # The function to be integrated is defined for the mutual information (MIN_DOUBLE is added to avoid divisions by 0).
            integrand0 = lambda a,b: f_xyz0_kde.evaluate([a,b]) * math.log((f_xyz0_kde.evaluate([a,b]) / ((f_xz0_kde.evaluate(a)*f_yz0_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)
            integrand1 = lambda a,b: f_xyz1_kde.evaluate([a,b]) * math.log((f_xyz1_kde.evaluate([a,b]) / ((f_xz1_kde.evaluate(a)*f_yz1_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)

            # The points to be integrated are defined.
            x_pts = np.linspace(data.iloc[:,nodo0].min()-data.iloc[:,nodo0].std(), data.iloc[:,nodo0].max()+data.iloc[:,nodo0].std(), precision)
            y_pts = np.linspace(data.iloc[:,nodo1].min()-data.iloc[:,nodo1].std(), data.iloc[:,nodo1].max()+data.iloc[:,nodo1].std(), precision)

            i = -1
            for x in x_pts:
                i += 1
                j = 0
                for y in y_pts:
                    integ0[i,j] = integrand0(x,y)[0]
                    integ1[i,j] = integrand1(x,y)[0] 
                    j += 1
            
            # The sum is calculated.
            summand = pz0 * np.trapz(np.trapz(integ0, x_pts, axis=0), y_pts, axis=0) + pz1 * np.trapz(np.trapz(integ1, x_pts, axis=0), y_pts, axis=0)

            # The value is saved in the matrix.
            functional_information[nodo0][nodo1] = float(summand)
            
        except Exception as e:
            print(f"Error occurred in node: {i}")
            print(datos1.iloc[:,nodo1])
            print(datos0.iloc[:,nodo1])
            print('Nodo0', nodo0)
            print('Nodo1', nodo1)
            print(f"Error message: {str(e)}")
            raise
    
    # If an URL is indicated the matrix is saved.

    if save != 'URL':

        df = pd.DataFrame(functional_information)
        df.to_csv(save)

    return functional_information



# """

# FUNCIONALIDAD

# La función calcula la información mutua entre aquellos pares micro-gen que tienen información estructural. Para ello, estima la distribución de cada elemento utilizando kernels gaussianos
# y definiendo el ancho de banda según la metodología indicada (dentro de las que scipy.stats.gaussian_kde tiene en cuenta). La función calcula la 
# información mutua integrando las funciones kernel, para ello se utiliza el método del trapecio, una metodología de integración numérica. La función 
# permite definir la precisión de la integración a través de la especificación del número de puntos en los que se evalúa la función.


# INPUTS

# datos: Diccionario con la expresión de los microRNAs y los genes en la key 'data' y sus clases en la key 'classvalues'. Es la salida que te ofrece la función cartesian_product_stratified().
# structural_information: Matriz con la información estructural.
# precision: Número de puntos en los que se evalua la función a la hora de calcular la integración numérica.


# OUTPUTS

# functional_information: Matriz de numpy con la información funcional entre cada par de nodos. La matriz será de dimensión (n+m)*(n+m) siendo n = nº de microRNA y m = nº de genes.


# """


# def mutual_info_trapz_matrix(dataset, structural_information, precision = 100, save = 'URL'):

#     data = dataset['data']

#     # Se define un valor muy pequeño para evitar la división por 0.
#     MIN_DOUBLE = 4.9406564584124654e-324 

#     nodos0 = np.where(structural_information != 0)[0]
#     nodos1 = np.where(structural_information != 0)[1]

#     # La proporción de elementos por clase (sólo válido para dos clases)
#     pz1 = dataset['classvalues'].sum() / len(dataset['classvalues'])
#     pz0 = 1 - pz1

#     # Se dividen los micros y los genes por clases
#     datos0 = data[(dataset['classvalues'] == 0).values]
#     datos1 = data[(dataset['classvalues'] == 1).values]

#     n_nodos = data.shape[1]
#     functional_information = np.zeros((n_nodos, n_nodos))

#     # Se definen los array para guardar los valores.
#     s = (precision,precision)
#     integ0 = np.zeros(s)
#     integ1 = np.zeros(s)

#     print('\nCALCULATING FUNCTIONAL MUTUAL INFORMATION...')
#     for i in tqdm(range(len(nodos0))):
#         try:
#             nodo0 = nodos0[i]
#             nodo1 = nodos1[i]
#             # Kernel asociado al microRNA de la conexión (P(X|Z)).
#             f_xz1_kde = gaussian_kde(datos1.iloc[:,nodo0])
#             f_xz0_kde = gaussian_kde(datos0.iloc[:,nodo0])

#             # Kernel asociado al gen de la conexión (P(Y|Z)).
#             f_yz1_kde = gaussian_kde(datos1.iloc[:,nodo1])
#             f_yz0_kde = gaussian_kde(datos0.iloc[:,nodo1])

#             # Kernel bidimensional con ambos elementos (P(X,Y|Z)).
#             f_xyz0_kde = gaussian_kde(np.vstack((datos0.iloc[:,nodo0], datos0.iloc[:,nodo1]))) 
#             f_xyz1_kde = gaussian_kde(np.vstack((datos1.iloc[:,nodo0], datos1.iloc[:,nodo1]))) 

#             # Se define la función a integrar de la información mutua (se suma el MIN_DOUBLE para evitar divisiones por 0).
#             integrand0 = lambda a,b: f_xyz0_kde.evaluate([a,b]) * math.log((f_xyz0_kde.evaluate([a,b]) / ((f_xz0_kde.evaluate(a)*f_yz0_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)
#             integrand1 = lambda a,b: f_xyz1_kde.evaluate([a,b]) * math.log((f_xyz1_kde.evaluate([a,b]) / ((f_xz1_kde.evaluate(a)*f_yz1_kde.evaluate(b)) + MIN_DOUBLE)) + MIN_DOUBLE)

#             # Se definen los puntos a integrar.
#             x_pts = np.linspace(data.iloc[:,nodo0].min()-data.iloc[:,nodo0].std(), data.iloc[:,nodo0].max()+data.iloc[:,nodo0].std(), precision)
#             y_pts = np.linspace(data.iloc[:,nodo1].min()-data.iloc[:,nodo1].std(), data.iloc[:,nodo1].max()+data.iloc[:,nodo1].std(), precision)

#             i = -1
#             for x in x_pts:
#                 i += 1
#                 j = 0
#                 for y in y_pts:
#                     integ0[i,j] = integrand0(x,y)[0]
#                     integ1[i,j] = integrand1(x,y)[0] 
#                     j += 1
            
#             # Se calcula el sumatorio.
#             summand = pz0 * np.trapz(np.trapz(integ0, x_pts, axis=0), y_pts, axis=0) + pz1 * np.trapz(np.trapz(integ1, x_pts, axis=0), y_pts, axis=0)

#             functional_information[nodo0][nodo1] = float(summand)
            
#         except Exception as e:
#             print(f"Error occurred in node: {i}")
#             print(datos1.iloc[:,nodo1])
#             print(datos0.iloc[:,nodo1])
#             print('Nodo0', nodo0)
#             print('Nodo1', nodo1)
#             print(f"Error message: {str(e)}")
#             raise
    
#     if save != 'URL':

#         df = pd.DataFrame(functional_information)
#         df.to_csv(save)

#     return functional_information