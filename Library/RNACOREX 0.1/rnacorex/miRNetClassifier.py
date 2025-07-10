import pandas as pd
from rnacorex import StrucInformation
from rnacorex import FuncInformation
from rnacorex import CLGStructure
from sklearn.base import BaseEstimator, ClassifierMixin


class MRNC(BaseEstimator, ClassifierMixin):
        
        """

        miRNetClassifier model.

        Attributes:

                n_con (int): The default number of connections of the model.
                precision (int): The precision used for the conditional mutual information approximation.
                mode (str): The mode used for the interaction ranking.
                weight (float): The weight used for the interaction ranking.

        """
    
        def __init__(self, n_con=20, precision = 10, mode = 'alternating', weight = 0.5):

            
            """
            
            Initializes a new instance of MRNC class.

            Args:

                    n_con (int): The default number of connections of the model.
                    precision (int): The precision used for the conditional mutual information approximation.
                    mode (str): The mode used for the interaction ranking.
                    weight (float): The weight used for the interaction ranking.

            """

            self.n_con = n_con

            self.precision = precision

            self.mode = mode

            self.weight = weight 


        def initialize_model(self, X, y):

            """

            Initializes the model with the data X and y.

            Args:
                
                    X (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y (pd.Series): The binary values of the class/phenotype.

            Returns:
                
                    self (MRNC): The initialized model.

            """

            self.struc_computed_ = False

            self.func_computed_ = False

            self.is_ranked_ = False

            self.is_fitted_ = False

            self.metrics_computed_ = False

            self.structural_information_, self.micros_, self.genes_, self.conex_, self.gtf = StrucInformation.run_engine_scikit(X, y)

            self.X_, self.y_ = X[self.micros_+self.genes_], y

            self.struc_computed_ = True

            return self
        


        def compute_functional(self, X_train = None, y_train = None):

            """
            
            Computes the functional information of the model.
            
            Args:
            
                    X_train (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_train (pd.Series): The binary values of the class/phenotype.
                    
            Returns:
                    
                    self (MRNC): The model with the computed functional information.
                    
            """

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before computing functional information.")
            
            if X_train is not None and y_train is not None:

                if not X_train.columns.equals(self.X_.columns):

                    raise ValueError("Columns of X_train do not match the columns of the initialized data X_.")

                else:

                    self.X_func = X_train

                    self.y_func = y_train

                    self.functional_information_ = FuncInformation.FMI2(self.X_func, self.y_func, self.structural_information_, precision = self.precision)
         
                    # self.functional_information_ = FuncInformation.mutual_info_trapz_matrix_scikit(self.X_func, self.y_func, self.structural_information_, precision = self.precision)

            else:

                self.functional_information_ = FuncInformation.FMI2(self.X_, self.y_, self.structural_information_, precision = self.precision)
                
                # self.functional_information_ = FuncInformation.mutual_info_trapz_matrix_scikit(self.X_, self.y_, self.structural_information_, precision = self.precision)

            self.func_computed_ = True

            return self
        


        def rank(self):

            """

            Ranks the interactions of the model based on the structural and functional information.
            
            Args:
                    None

            Returns:
                    self (MRNC): The model with the ranked interactions.

            """

            self.intern_connections_ = CLGStructure.order_interactions(self.structural_information_, self.functional_information_, self.mode, self.weight)

            self.connections_ = CLGStructure.get_ranking(self.micros_+self.genes_, self.intern_connections_, self.gtf)

            self.is_ranked_ = True

            return self
        
    

        def fit_only(self, new_sets = False):

            """
            
            Fits an initialized model with previously computed structural and functional information.
            
            Args:
            
                    new_sets (bool): If True, the model will be fitted with new sets of data. If False, it will be fitted with the previously computed data.
                    
            Returns:
                    
                    self (MRNC): The fitted model.
                    
            """

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before computing functional information.")
            
            if not self.func_computed_:

                raise RuntimeError("Functional information is not computed. Call 'compute_functional' before executing the interaction ranking.")
            
            # if X_train is None and y_train is None:

            #     X_train, y_train = self.X_, self.y_

            if new_sets:

                self.nodos_dag_, self.clgc_ = CLGStructure.fit_model(self.X_func, self.y_func, self.n_con, self.intern_connections_)
            
            else:

                self.nodos_dag_, self.clgc_ = CLGStructure.fit_model(self.X_, self.y_, self.n_con, self.intern_connections_)

            self.is_fitted_ = True

            return self
        
        
        
        def fit(self, X_train, y_train):

            
            """

            Fits a non initialized model with the X_train and y_train data.

            Args:

                    X_train (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_train (pd.Series): The binary values of the class/phenotype.

            Returns:
                
                    self (MRNC): The fitted model.
    
            """

            self.struc_computed_ = False

            self.func_computed_ = False

            self.is_ranked_ = False

            self.is_fitted_ = False

            if not self.struc_computed_:

                self.structural_information_, self.micros_, self.genes_, self.conex_, self.gtf = StrucInformation.run_engine_scikit(X_train, y_train)

                self.X_, self.y_ = X_train[self.micros_+self.genes_], y_train

                self.struc_computed_ = True
            
            if not self.func_computed_:

                self.functional_information_ = FuncInformation.FMI2(self.X_, self.y_, self.structural_information_, precision = self.precision)

                # self.functional_information_ = FuncInformation.mutual_info_trapz_matrix_scikit(self.X_, self.y_, self.structural_information_, precision = self.precision)

                self.func_computed_ = True

            if not self.is_ranked_:
            
                self.intern_connections_ = CLGStructure.order_interactions(self.structural_information_, self.functional_information_, self.mode, self.weight)

                self.connections_ = CLGStructure.get_ranking(self.micros_+self.genes_, self.intern_connections_, self.gtf)

                self.is_ranked_ = True

            self.nodos_dag_, self.clgc_ = CLGStructure.fit_model(self.X_, self.y_, self.n_con, self.intern_connections_)

            self.is_fitted_ = True

            return self
        


        def structure_search(self, X_train = None, y_train = None, X_test = None, y_test = None, max_models = None):

            """
            
            Searches for the structure of the model.
            
            Args:

                    X_train (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_train (pd.Series): The binary values of the class/phenotype.
                    X_test (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_test (pd.Series): The binary values of the class/phenotype.
                    max_models (int): The maximum number of models to be computed.
                    
            Returns:
                    
                    self (MRNC): The model with the computed structure metrics.
                    metrics (pd.DataFrame): The computed structure metrics.
                    
            """


            self.metrics_computed_ = False

            if X_train is None and y_train is None:

                if X_test is None and y_test is None:

                    # Without TRAIN

                    if hasattr(self, 'intern_connections_'):

                        self.structure_metrics_ = CLGStructure.structure_search(X_train = self.X_, y_train = self.y_, max_models = max_models, conexiones = self.intern_connections_)

                        self.metrics_computed_ = True

                    else:

                        raise RuntimeError("Model has not been initialized. Please initialize.")
                    
                else:
                    
                    if hasattr(self, 'intern_connections_'):

                        X_test = X_test[self.micros_+self.genes_]

                        self.structure_metrics_ = CLGStructure.structure_search(X_train = self.X_, y_train = self.y_, X_test = X_test, y_test = y_test, max_models = max_models, conexiones = self.intern_connections_)

                        self.metrics_computed_ = True

                    else:

                        raise RuntimeError("Model has not been initialized. Please initialize.")
            
            else:

                if X_test is None and y_test is None:

                    if hasattr(self, 'intern_connections_'):

                        self.structure_metrics_ = CLGStructure.structure_search(X_train = X_train, y_train = y_train, max_models = max_models, conexiones = self.intern_connections_)

                        self.metrics_computed_ = True

                    else:

                        raise RuntimeError("Model has not been initialized. Please initialize.")
                
                else:

                    if hasattr(self, 'intern_connections_'):

                        X_test = X_test[self.micros_+self.genes_]

                        X_train = X_train[self.micros_+self.genes_]

                        self.structure_metrics_ = CLGStructure.structure_search(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, max_models = max_models, conexiones = self.intern_connections_)

                        self.metrics_computed_ = True

                    else:

                        raise RuntimeError("Model has not been initialized. Please initialize.")

            return self.structure_metrics_
    



        def predict(self, X_test):

            """
            
            Predicts the class of the input data X_test.
            
            Args:
            
                    X_test (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    
            Returns:
                    
                    classification (np.array): The predicted class of the input data X_test.
                    
            """

            if not self.is_fitted_:

                raise RuntimeError("Estimator is not fitted. Call 'fit' before exploiting the model.")

            # Only those miRNAs and genes that have been used in the training will be used.

            X_test = X_test[self.micros_+self.genes_]

            return CLGStructure.predict_test(X_test, self.nodos_dag_, self.clgc_)['classification']
        


        def predict_proba(self, X_test):

            """
            
            Predicts the class probabilities of the input data X_test.
            
            Args:
            
                    X_test (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    
            Returns:
                    
                    classification (np.array): The predicted class probabilities of the input data X_test.
                    
            """

            if not self.is_fitted_:

                raise RuntimeError("Estimator is not fitted. Call 'fit' before exploiting the model.")
            
            # Only those miRNAs and genes that have been used in the training will be used.

            X_test = X_test[self.micros_+self.genes_]

            return CLGStructure.predict_test(X_test, self.nodos_dag_, self.clgc_)['posterior']
        

        

        def get_network(self, k = None, display = False):

            """
            
            Shows the network of the model.
            
            Args:
            
                    k (int): The number of connections to be shown.
                    display (bool): If True, the network will be displayed.
                    
            Returns:
                    
                    self (MRNC): The model with the computed structure metrics.
                    
            """

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before showing connections.")

            if not self.is_ranked_:

                raise RuntimeError("Interaction ranking is not computed. Call 'interaction_ranking' before showing connections.")

            if k == None:

                self.G_ = CLGStructure.get_network(self.connections_, self.n_con, self.gtf, display)
            
            else:

                self.G_ = CLGStructure.get_network(self.connections_, k, self.gtf, display)
        


        
