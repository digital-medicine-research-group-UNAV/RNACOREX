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

    """
    Predicts class labels and posterior probabilities for test samples
    using a collection of fitted probabilistic network models.

    For each model, checks whether all required features are available
    in the test data. If evidence is incomplete, performs inference using
    the observed variables when supported by the model type. Otherwise,
    computes class-conditional likelihoods using the appropriate model-
    specific probability density function and combines them with class
    prior probabilities to obtain posterior probabilities.

    Args:
        X_test (pd.DataFrame): Test dataset containing samples as rows
            and feature names as columns.
        clgc (dict): Dictionary of fitted models, each containing model
            parameters, feature labels, class prior probabilities, and
            model-specific information.

    Returns:
        dict: Dictionary containing the prediction results with the
        following keys:
            - 'posterior' (list): Posterior class probabilities for each
              model and test sample.
            - 'classification' (list): Predicted class indices for each
              model and test sample.
            - 'k' (list): Keys identifying the models used for prediction.
    """

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

            # BREAK IF MODEL = 'SLGBN' OR MODEL = 'SPBN'

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

                # Residual error covariance matrix.
                D_c = np.diag(var_c)

                # Model induced covariance matrix
                A_inv = np.linalg.solve(A, np.eye(n))
                Sigma_c = A_inv @ D_c @ A_inv.T

                # Select observed variables
                Sigma_oo = Sigma_c[np.ix_(obs_idx, obs_idx)]
                mu_o = mu_c[obs_idx]

                class_params.append((mu_o, Sigma_oo))

            # Calculate likelihoods
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

            # Normalize using log-sum-exp
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

    """
    Computes the Maximum Probability Configuration (MPC) for each class in a
    Linear Gaussian Bayesian Network (LGBN) and optionally visualizes
    class-specific deviations from a global reference.

    Calculates the expected value of each node for every class by solving
    the linear system defined by the model parameters. When visualization
    is enabled, displays the network with nodes colored according to their
    standardized deviations from either a model-implied or empirical
    reference.

    Args:
        params (dict): Fitted model parameters, including the model type,
            class probabilities ('classp'), regression coefficients
            ('beta'), and residual variances ('sigma').
        display (bool): Whether to display the network visualization.
        c (int or None): Index of the class to visualize.
        G (nx.Graph or None): NetworkX graph representing the network.
        th (float, optional): Absolute threshold defining the minimum and
            maximum values of the color scale, centered at zero.
            Defaults to 1.
        figsize (tuple, optional): Figure dimensions in inches.
            Defaults to (8, 8).
        X (np.ndarray or None, optional): Observed data with shape
            (n_samples, n_nodes), required when reference is 'empirical'.
            Defaults to None.
        reference (str, optional): Reference used to standardize class
            deviations. Options are 'model' and 'empirical'.
            Defaults to 'model'.
        save_format (str or None, optional): Format used to save the
            visualization, such as 'png', 'pdf', or 'svg'. If None, the
            figure is not saved. Defaults to None.
        ax (matplotlib.axes.Axes or None, optional): Existing Matplotlib
            axes on which to draw the network. If None, a new figure and
            axes are created. Defaults to None.

    Returns:
        np.ndarray or None: Array of shape (n_classes, n_nodes)
        containing the expected node values for each class. Returns None
        if the model is not an LGBN.
    """

    if params['model'] != 'LGBN':

        print('Model: ', params['model'])

        print('MPC is only available with LGBN model.')

        return
    
    if reference not in ['model', 'empirical']:
        raise ValueError(
            "reference must be either 'model' or 'empirical'."
        )

    # Class prior
    classp = params['classp']

    betas = params['beta']

    sigmas = params['sigma']

    n_nodes, _, n_classes = betas.shape

    maps = np.zeros((n_classes, n_nodes))
    
    for i in range(n_classes):

        # Intercepts
        mu = np.diag(betas[:,:,i]) 

        # Coefficients
        B = betas[:,:,i].copy()

        np.fill_diagonal(B, 0)    

        X_map = np.linalg.solve(np.eye(n_nodes) - B, mu)

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

                global_mean = np.dot(classp, maps)

                sigma2 = sigmas.T

                global_var = np.sum(
                    classp[:, None] * (
                        sigma2 + (maps - global_mean[None, :])**2
                    ),
                    axis=0
                )

                global_sigma = np.sqrt(global_var)

                reference_label = "model-implied reference"


            elif reference == 'empirical':

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

                global_mean = np.mean(X, axis=0)

                global_sigma = np.std(X, axis=0, ddof=1)

                reference_label = "empirical reference"

            class_mean = maps[c]

            z = (class_mean - global_mean) / (global_sigma + 1e-9)  
            colors = ["lightcoral", "skyblue", "lightgreen"]
            cmap = LinearSegmentedColormap.from_list("custom_map", colors)

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