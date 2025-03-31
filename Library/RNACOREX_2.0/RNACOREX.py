import StrucInformation
import InitModel as InitModel
import FuncInformation
import CLGStructure as CLGStructure
import copy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin



class RNACOREX(BaseEstimator, ClassifierMixin):

    def __init__(self, n_con = 20, precision = 10, mirna = True, mrna = True, lncrna = True, engine = None):

        self.precision = precision

        self.n_con = n_con
        
        self.mirna = mirna

        self.mrna = mrna

        self.lncrna = lncrna

        self.engine = engine
    

    def fit(self, X, y):

        engine = self.engine

        self.X_, self.filtered_engine_, self.info_ = InitModel.filter_db(X, mrna = self.mrna, mirna = self.mirna, lncrna = self.lncrna, engine = engine)

        self.X_, self.SMI_ = StrucInformation.SMI(self.X_, self.info_, self.filtered_engine_)

        self.FMI_ = FuncInformation.FMI2(self.X_, y, self.SMI_, precision = self.precision)

        # Define how to develop the model (single n, list or tuple)

        if isinstance(self.n_con, tuple) and len(self.n_con) == 2:

            n_con_values = list(range(self.n_con[0], self.n_con[1] + 1))

        elif isinstance(self.n_con, list):

            n_con_values = self.n_con
        
        else:

            n_con_values = [self.n_con]
        
        self.models_ = {n: CLGStructure.fit_model(X, y, copy.deepcopy(self.SMI_), copy.deepcopy(self.FMI_), n, copy.deepcopy(self.info_)) for n in tqdm(n_con_values, desc="Fitting models", unit="model")}

        return self
    

    def predict(self, X):

        rl = CLGStructure.predict_test(X, self.models_)

        return rl
    
    def get_interactions(self, n):

        interactions = CLGStructure.get_interactions(self.SMI_, self.FMI_, self.info_, n)

        return interactions