from . import StrucInformation
from . import InitModel
from . import FuncInformation
from . import CLGStructure
from . import Functionalities
from . import Inference
# import StrucInformation
# import InitModel
# import FuncInformation
# import CLGStructure
# import Functionalities
# import Inference
import copy
import time
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin



class RNACOREX(BaseEstimator, ClassifierMixin):

    def __init__(self, n_con = 20, precision = 10, model = 'LGBN', mirna = True, mrna = True, lncrna = True, engine = None, mode = 'alternating', weight = 0.5, ties = 'isolated'):

        self.n_con = n_con
        
        self.precision = precision

        self.model = model
        
        self.mirna = mirna

        self.mrna = mrna

        self.lncrna = lncrna

        self.engine = engine

        self.mode = mode

        self.weight = weight

        self.ties = ties
    

    def fit(self, X, y):

        engine = self.engine

        start_initialize = time.perf_counter()

        self.X_, self.filtered_engine_, self.info_ = InitModel.filter_db(X, mrna = self.mrna, mirna = self.mirna, lncrna = self.lncrna, engine = engine)

        self.X_, self.SMI_ = StrucInformation.SMI(self.X_, self.info_, self.filtered_engine_)

        end_initialize = time.perf_counter()

        self.initialize_time_ = end_initialize-start_initialize

        start_FMI = time.perf_counter()

        self.FMI_ = FuncInformation.FMI2(self.X_, y, self.SMI_, precision = self.precision)

        end_FMI = time.perf_counter()

        self.FMI_time_ = end_FMI-start_FMI

        # Define how to develop the model (single n, list or tuple)

        if isinstance(self.n_con, tuple) and len(self.n_con) == 2:

            n_con_values = list(range(self.n_con[0], self.n_con[1] + 1))

        elif isinstance(self.n_con, list):

            n_con_values = self.n_con
        
        else:

            n_con_values = [self.n_con]

        start_rank = time.perf_counter()

        ranking = CLGStructure.build_ranking(
            self.SMI_,
            self.FMI_,
            mode=self.mode,
            weight=self.weight,
            ties=self.ties
        )

        end_rank = time.perf_counter()

        self.rank_time_ = end_rank-start_rank

        start_fit = time.perf_counter()

        # self.models_ = {n: CLGStructure.fit_model(X, y, copy.deepcopy(self.SMI_), copy.deepcopy(self.FMI_), n, copy.deepcopy(self.model), copy.deepcopy(self.mode), copy.deepcopy(self.weight), copy.deepcopy(self.ties), copy.deepcopy(self.info_)) for n in tqdm(n_con_values, desc="Fitting models", unit="model")}

        # self.models_ = {n: CLGStructure.fit_model(X, y, copy.deepcopy(self.SMI_), copy.deepcopy(self.FMI_), n, copy.deepcopy(ranking), copy.deepcopy(self.model), copy.deepcopy(self.info_)) for n in tqdm(n_con_values, desc="Fitting models", unit="model")}

        self.models_ = {n: CLGStructure.fit_model(X, y, self.SMI_, self.FMI_, n, ranking, self.model, copy.deepcopy(self.info_)) for n in tqdm(n_con_values, desc="Fitting models", unit="model")}

        end_fit = time.perf_counter()

        self.fit_time_mean_ = (end_fit - start_fit) / len(n_con_values)

        if len(self.models_) == 1:

            self.model_ = next(iter(self.models_.values()))

        else:
            
            self.model_ = None

        return self
    

    def mpc(self, display = False, c = None, k = None, th = 1, figsize=(8, 8), reference='model', X=None, save_format=None, ax=None):

        mpc_values = {n: Inference.mpc(self.models_[n], display=display and (n == k), c=c, G=self.models_[n]['G'] if n == k else None, th=th, figsize=figsize, reference = reference, X = X, save_format=save_format, ax=ax) for n in self.models_}

        return mpc_values


    def predict(self, X):

        rl = Inference.predict_test(X, self.models_)

        return rl
    
    
    def get_interactions(self, n):

        interactions = Functionalities.get_interactions(self.SMI_, self.FMI_, self.mode, self.weight, self.ties, self.info_, n)

        return interactions
    
    def display(self, G, save_format=None):

        Functionalities.display(G, save_format=save_format)

        return self