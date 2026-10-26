import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
from scipy.stats import gaussian_kde


def compute_mutual_info(
    nodo0,
    nodo1,
    X_train,
    y_train,
    precision,
    MIN_DOUBLE=4.9406564584124654e-324,
):
    """
    Calcula la información mutua condicional asociada a un par de nodos,
    utilizando KDE y una cuadrícula vectorizada.

    X_train: ndarray de forma (n_muestras, n_variables)
    y_train: ndarray de forma (n_muestras,)
    """

    # Evitar valores numéricos exactamente iguales a cero.
    MIN_DOUBLE = np.finfo(np.float64).tiny

    # Extraer las dos variables una sola vez.
    x = X_train[:, nodo0]
    y = X_train[:, nodo1]

    # Obtener las clases y sus probabilidades.
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_probabilities = counts / counts.sum()

    # Cuadrículas de integración.
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    x_pts = np.linspace(
        np.min(x) - x_std,
        np.max(x) + x_std,
        precision,
    )
    y_pts = np.linspace(
        np.min(y) - y_std,
        np.max(y) + y_std,
        precision,
    )

    # Preparar la cuadrícula bidimensional una sola vez.
    xx, yy = np.meshgrid(x_pts, y_pts, indexing="ij")
    points = np.vstack((xx.ravel(), yy.ravel()))

    summand = 0.0

    for z_value, pz in zip(unique_classes, class_probabilities):

        mask = y_train == z_value

        x_z = x[mask]
        y_z = y[mask]

        # Crear las KDE de cada clase.
        f_xz_kde = gaussian_kde(x_z)
        f_yz_kde = gaussian_kde(y_z)
        f_xyz_kde = gaussian_kde(np.vstack((x_z, y_z)))

        # Evaluar las densidades marginales una vez por eje.
        f_xz = f_xz_kde.evaluate(x_pts)
        f_yz = f_yz_kde.evaluate(y_pts)

        # Evaluar toda la KDE conjunta en una sola llamada.
        f_xyz = f_xyz_kde.evaluate(points).reshape(
            len(x_pts),
            len(y_pts),
        )

        # Producto de densidades marginales.
        denominator = (
            f_xz[:, None] * f_yz[None, :] + MIN_DOUBLE
        )

        # Integrando de la información mutua.
        integ = f_xyz * np.log(
            f_xyz / denominator + MIN_DOUBLE
        )

        # Integración numérica, conservando el orden original.
        integral = np.trapz(
            np.trapz(integ, x_pts, axis=0),
            y_pts,
            axis=0,
        )

        summand += pz * integral

    return nodo0, nodo1, float(summand)


def FMI2(
    X_train,
    y_train,
    structural_information,
    precision,
    n_jobs=-1,
    inner_max_num_threads=1,
):
    """
    Calcula la matriz de información funcional mediante KDE paralelizada.

    Parámetros
    ----------
    X_train : pandas.DataFrame o ndarray
        Datos de entrenamiento.
    y_train : pandas.Series o ndarray
        Etiquetas de clase.
    structural_information : ndarray
        Matriz que determina las conexiones que deben procesarse.
    precision : int
        Número de puntos por eje de la cuadrícula KDE.
    n_jobs : int
        Número de trabajos de joblib. Usa 1, 2, 4 o -1.
    inner_max_num_threads : int o None
        Límite de hilos internos de BLAS/OpenMP para trabajadores loky.
    """

    # Convertir los datos a NumPy una sola vez.
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)

    if X.ndim != 2:
        raise ValueError("X_train debe ser una matriz bidimensional.")

    if y.ndim != 1:
        raise ValueError("y_train debe ser un vector unidimensional.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X_train e y_train deben tener el mismo número de muestras."
        )

    if precision < 2:
        raise ValueError("precision debe ser al menos 2.")

    structural_information = np.asarray(structural_information)

    if structural_information.shape != (
        X.shape[1],
        X.shape[1],
    ):
        raise ValueError(
            "La dimensión de structural_information debe coincidir "
            "con el número de variables de X_train."
        )

    # Obtener las conexiones estructurales.
    nodos0, nodos1 = np.where(structural_information != 0)

    # Inicializar la matriz de salida.
    functional_information = np.zeros_like(
        structural_information,
        dtype=np.float64,
    )

    # Configurar el paralelismo.
    parallel_kwargs = {"n_jobs": n_jobs}

    if n_jobs != 1 and inner_max_num_threads is not None:
        with parallel_config(
            backend="loky",
            inner_max_num_threads=inner_max_num_threads,
        ):
            results = Parallel(**parallel_kwargs)(
                delayed(compute_mutual_info)(
                    nodo0,
                    nodo1,
                    X,
                    y,
                    precision,
                )
                for nodo0, nodo1 in tqdm(
                    zip(nodos0, nodos1),
                    total=len(nodos0),
                    desc="Calculating mutual information",
                )
            )
    else:
        results = Parallel(**parallel_kwargs)(
            delayed(compute_mutual_info)(
                nodo0,
                nodo1,
                X,
                y,
                precision,
            )
            for nodo0, nodo1 in tqdm(
                zip(nodos0, nodos1),
                total=len(nodos0),
                desc="Calculating mutual information",
            )
        )

    # Rellenar la matriz con los resultados.
    if len(results) > 0:
        result_nodos0, result_nodos1, result_values = zip(*results)

        functional_information[
            np.asarray(result_nodos0),
            np.asarray(result_nodos1),
        ] = np.asarray(result_values)

    return functional_information