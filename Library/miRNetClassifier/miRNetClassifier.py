import pandas as pd
import StrucInformation
import FuncInformation
import CLGStructure
from sklearn.base import BaseEstimator, ClassifierMixin


class MRNC(BaseEstimator, ClassifierMixin):
        
        """

        A class to represent a person.

        Attributes:

                n_con (int): The default number of connections of the model.
                precision (int): The precision used for the conditional mutual information approximation.

        """
    
        def __init__(self, n_con=20, precision = 10):

            
            """
            
            Initializes a new instance of MRNC class.

            Args:

                    n_con (int): The default number of connections of the model.
                    precision (int): The precision used for the conditional mutual information approximation.

            """

            self.n_con = n_con

            self.precision = precision


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

            self.structural_information_, self.micros_, self.genes_, self.gtf = StrucInformation.run_engine_scikit(X, y)

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
            
            if X_train is None and y_train is None:

                X_train, y_train = self.X_, self.y_

            else:

                if not X_train.columns.equals(self.X_.columns):

                    raise ValueError("Columns of X_train do not match the columns of the initialized data X_.")
         
            self.functional_information_ = FuncInformation.mutual_info_trapz_matrix_scikit(X_train, y_train, self.structural_information_, precision = self.precision)

            self.func_computed_ = True

            return self
    

        def fit_only(self):

            """
            
            Fits an initialized model with previously computed structural and functional information.
            
            Args:
            
                    X_train (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_train (pd.Series): The binary values of the class/phenotype.
                    
            Returns:
                    
                    self (MRNC): The fitted model.
                    
            """

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before computing functional information.")
            
            if not self.func_computed_:

                raise RuntimeError("Functional information is not computed. Call 'compute_functional' before executing the interaction ranking.")

            self.intern_connections_ = CLGStructure.order_interactions(self.structural_information_, self.functional_information_)

            self.connections_ = CLGStructure.get_ranking(self.micros_+self.genes_, self.intern_connections_, self.gtf)

            self.is_ranked_ = True

            if self.is_ranked_ is None:

                raise RuntimeError("Interaction ranking is not computed. Call 'interaction_ranking' before fitting.")
            
            # if X_train is None and y_train is None:

            #     X_train, y_train = self.X_, self.y_

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

                self.structural_information_, self.micros_, self.genes_, self.gtf = StrucInformation.run_engine_scikit(X_train, y_train)

                self.X_, self.y_ = X_train[self.micros_+self.genes_], y_train

                self.struc_computed_ = True
            
            if not self.func_computed_:

                self.functional_information_ = FuncInformation.mutual_info_trapz_matrix_scikit(self.X_, self.y_, self.structural_information_, precision = self.precision)

                self.func_computed_ = True

            if not self.is_ranked_:
            
                self.intern_connections_ = CLGStructure.order_interactions(self.structural_information_, self.functional_information_)

                self.connections_ = CLGStructure.get_ranking(self.micros_+self.genes_, self.intern_connections_, self.gtf)

                self.is_ranked_ = True

            self.nodos_dag_, self.clgc_ = CLGStructure.fit_model(X_train, y_train, self.n_con, self.intern_connections_)

            self.is_fitted_ = True

            return self
        


        def structure_search(self, X_train = None, y_train = None, X_test = None, y_test = None, max_models = None, link_txt = False):

            """
            
            Searches the structure of the model.
            
            Args:
            
                    X_train (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_train (pd.Series): The binary values of the class/phenotype.
                    X_test (pd.DataFrame): The input data with miRNA and mRNA expressions.
                    y_test (pd.Series): The binary values of the class/phenotype.
                    max_models (int): The maximum number of models to be computed.
                    link_txt (str): A string with a url to the .txt document in which the information want to be saved.
                    
            Returns:
                    
                    self (MRNC): The model with the computed structure metrics.
                    
            """

            self.metrics_computed_ = False

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before computing functional information.")
            
            if not self.func_computed_:

                raise RuntimeError("Functional information is not computed. Call 'compute_functional' before executing the interaction ranking.")

            self.intern_connections_ = CLGStructure.order_interactions(self.structural_information_, self.functional_information_)

            if X_train is None and y_train is None:

                X_train = self.X_

                y_train = self.y_

            if X_test is None and y_test is None:

                print('Models are calculated without test data.')

            else:
                    
                X_test = X_test[self.micros_+self.genes_]

            self.structure_metrics_ = CLGStructure.structure_search(X_train, y_train, X_test, y_test, max_models, self.intern_connections_, link_txt)

            self.metrics_computed_ = True

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

            # Para utilizar sólo aquellos micros y genes que se han utilizado en el entrenamiento.

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
            
            # Para utilizar sólo aquellos micros y genes que se han utilizado en el entrenamiento.

            X_test = X_test[self.micros_+self.genes_]

            return CLGStructure.predict_test(X_test, self.nodos_dag_, self.clgc_)['posterior']
        

        

        def get_network(self, k = None, save = 'url'):

            """
            
            Shows the network of the model.
            
            Args:
            
                    k (int): The number of connections to be shown.
                    save (str): The path to save the network.
                    
            Returns:
                    
                    self (MRNC): The model with the computed structure metrics.
                    
            """

            if not self.struc_computed_:

                raise RuntimeError("Structural information is not computed. Call 'initialize_model' before showing connections.")

            if not self.is_ranked_:

                raise RuntimeError("Interaction ranking is not computed. Call 'interaction_ranking' before showing connections.")

            if k == None:

                CLGStructure.get_network(self.connections_, self.n_con, save = save)
            
            else:

                CLGStructure.get_network(self.connections_, k, save = save)
        


        