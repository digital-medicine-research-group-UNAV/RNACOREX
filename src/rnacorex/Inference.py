from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from .models.SLGBN import pdf_slgbn
from .models.SPBN import logpdf_spbn
from .models.LGBN import pdf_lgbn
# from models.SLGBN import pdf_slgbn
# from models.SPBN import logpdf_spbn
# from models.LGBN import pdf_lgbn
import networkx as nx


def predict_test(X_test, clgc):

    classifications = []
    posteriors = []
    claves = []

    for clave in tqdm(clgc, desc="Predicting", unit="model"):

        model = clgc[clave]
        ensgs = clgc[clave]['labels'] 
        entries = X_test.columns.tolist()


        missing = set(ensgs) - set(entries)

        if missing:

            if model['model'] in ['SLGBN', 'SPBN']:

                print('Inference with partial evidence is not allowed with SLGBN or SPBN models.')

                break

            # BREAK THIS PART IF MODEL = 'SLGBN' OR MODEL = 'SPBN'

            print('K:', clave, ' Predicting with partial evidence.')

            params = model

            # Extract network information
            dag = params['dag']        # dag[parent, child] == 1
            beta = params['beta']      # beta[child, parent, class]
            sigma = params['sigma']    # residual variances, shape (n, k)
            classp = params['classp']  # shape (k,)
            labels = params['labels']

            cont_nodes = [n for n in labels if n != 'C']

            node_index = {n:i for i,n in enumerate(cont_nodes)}

            n = len(cont_nodes)
            k = len(classp)

            obs_nodes = [node for node in cont_nodes if node in X_test.columns]
            obs_idx = [node_index[n] for n in obs_nodes]

            X = X_test[obs_nodes].to_numpy()
            m = X.shape[0]

            posterior = np.zeros((m, k))
            class_params = []

            # Precompute class-specific mean/covariance
            for c in range(k):
                B = np.zeros((n, n))
                b = np.zeros(n)
                for i in range(n):
                    b[i] = beta[i, i, c]
                    for j in range(n):
                        # dag[j, i] significa que j es padre de i.
                        if dag[j, i] == 1:
                            B[i, j] = beta[i, j, c]
                A = np.eye(n) - B

                mu_c = np.linalg.solve(A, b)
                var_c = sigma[:, c]

                # Matriz de covarianzas de los errores residuales.
                D_c = np.diag(var_c)

                # Covarianza conjunta inducida por la red.
                A_inv = np.linalg.solve(A, np.eye(n))
                Sigma_c = A_inv @ D_c @ A_inv.T

                # Seleccionar las variables observadas.
                Sigma_oo = Sigma_c[np.ix_(obs_idx, obs_idx)]
                mu_o = mu_c[obs_idx]

                class_params.append((mu_o, Sigma_oo))

            # Calcular likelihoods
            log_posterior = np.zeros((m, k))

            for c in range(k):

                mu_obs, Sigma_obs = class_params[c]

                if len(obs_idx) == 0:
                    log_likelihood = np.zeros(m)
                else:
                    log_likelihood = multivariate_normal.logpdf(
                        X,
                        mean=mu_obs,
                        cov=Sigma_obs,
                        allow_singular=True
                    )

                log_prior = np.log(np.clip(classp[c], 1e-300, None))

                log_posterior[:, c] = log_prior + log_likelihood

            # Normalizar utilizando log-sum-exp
            log_norm = logsumexp(log_posterior, axis=1, keepdims=True)

            posterior = np.exp(log_posterior - log_norm)

            maxclass = np.argmax(posterior, axis=1)

            posteriors.append(posterior)
            classifications.append(maxclass)
            claves.append(clave)


        else:

            print('K:', clave, ' Predicting with complete evidence.')
            
            dataset_te = X_test[model['labels']]

            m = X_test.shape[0]

            k = len(model['classp'])

            priorprob = model["classp"]

            if model['model'] == 'SPBN':

                log_condpdfx = np.zeros((m,k))

                for c in range(0,k):

                    log_condpdfx[:,c] = logpdf_spbn(dataset_te, model['params'][c])

                log_priorprob = np.log(np.clip(priorprob, 1e-300, None))
                log_posterior = log_condpdfx + log_priorprob[np.newaxis, :]
                log_norm = logsumexp(log_posterior, axis=1, keepdims=True)
                posterior = np.exp(log_posterior - log_norm)

            else:

                condpdfx = np.zeros((m,k))
                    
                for c in range(0,k):

                    if model['model'] == 'LGBN':

                        condpdfx[:,c] = pdf_lgbn(dataset_te, model['beta'][:,:,c], model['sigma'][:,c], model['dag'])

                    elif model['model'] == 'SLGBN':

                        condpdfx[:,c] = pdf_slgbn(dataset_te, model['params'][c], model['dag'])

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


def mpc(params, display, c, G, th = 1, figsize=(8, 8), X = None, reference='model', save_format=None, ax=None):

    if params['model'] != 'LGBN':

        print('Model: ', params['model'])

        print('MPC is only available with LGBN model.')

        return
    
    if reference not in ['model', 'empirical']:
        raise ValueError(
            "reference must be either 'model' or 'empirical'."
        )

    # Proporción de clases
    classp = params['classp']

    # Parámetros
    betas = params['beta']

    # Varianzas
    sigmas = params['sigma']

    n_nodes, _, n_classes = betas.shape

    maps = np.zeros((n_classes, n_nodes))
    
    for i in range(n_classes):

        # Interceptos
        mu = np.diag(betas[:,:,i]) 

        # Parámetros
        B = betas[:,:,i].copy()

        # Eliminar los interceptos de los parámetros
        np.fill_diagonal(B, 0)    

        # Resolver el sistema (I-B)X = mu
        X_map = np.linalg.solve(np.eye(n_nodes) - B, mu)

        # Valor esperado (y MPC)
        maps[i] = X_map

    if display:

        if G is None:
            print("Please select network for display.")

        elif c is None:
            print("Class index c is required for display.")

        else:

            import matplotlib.pyplot as plt
            from networkx.drawing.nx_agraph import graphviz_layout
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            pos = graphviz_layout(G, prog='neato')

            if reference == 'model':

                # Media global por variable (ambas clases)
                global_mean = np.dot(classp, maps)

                # Varianza por variable y clase
                sigma2 = sigmas.T

                # Construir varianza general (utilizando varianza de cada clase) por variable.
                global_var = np.sum(
                    classp[:, None] * (
                        sigma2 + (maps - global_mean[None, :])**2
                    ),
                    axis=0
                )

                # Obtener desviación estándar global
                global_sigma = np.sqrt(global_var)

                reference_label = "model-implied reference"


            elif reference == 'empirical':

                # Datos observados necesarios para la referencia empírica
                if X is None:
                    raise ValueError(
                        "X must be provided when reference='empirical'."
                    )

                X = np.asarray(X)

                if X.ndim != 2:
                    raise ValueError(
                        "X must be a 2D array with shape "
                        "(n_samples, n_nodes)."
                    )

                if X.shape[1] != n_nodes:
                    raise ValueError(
                        f"X has {X.shape[1]} variables, but the model "
                        f"contains {n_nodes} nodes."
                    )

                # Media empírica por variable
                global_mean = np.mean(X, axis=0)

                # Desviación estándar empírica por variable
                global_sigma = np.std(X, axis=0, ddof=1)

                reference_label = "empirical reference"

            class_mean = maps[c]

            # Normalización de los valores MPC con varianza especifica de cada variable (general, no por clase)
            z = (class_mean - global_mean) / (global_sigma + 1e-9)  
            colors = ["lightcoral", "skyblue", "lightgreen"]
            cmap = LinearSegmentedColormap.from_list("custom_map", colors)

            # Normalización centrada en 0 para colormap divergente
            norm_obj = TwoSlopeNorm(vmin=-th, vcenter=0, vmax=th)
            node_colors = [cmap(norm_obj(v)) for v in z]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)

            sm = ScalarMappable(norm=norm_obj, cmap=cmap)
            sm.set_array([])  # Required for matplotlib < 3.6
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, fraction=0.02)
            cbar.set_label("Z-score vs weighted mean", fontsize=12)

            ax.axis('off')
            ax.set_title(f'Class {c} vs global reference', fontsize=14)

            if save_format is not None:

                filename = f"mpc_{c}.{save_format}"
                fig.savefig(
                    filename,
                    format=save_format,
                    bbox_inches='tight'
                )
                print(f"Figure saved as: {filename}")

            # plt.show()
        
    return maps