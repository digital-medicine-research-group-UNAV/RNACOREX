import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import rnacorex

start_time = time.time()

# rnacorex.download()

rnacorex.check_engines()

def GNN(G_base, X_train, X_test, y_train, y_test, model = 'GCN'):

    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.utils import from_networkx


    num_nodes = G_base.number_of_nodes()
    edge_index = from_networkx(G_base).edge_index
    num_feats = 1 

    # --- Paso 2: Crear función para transformar una fila en Data ---
    def row_to_data(row, edge_index, y=None):
        x = torch.tensor(row.values, dtype=torch.float).view(num_nodes, num_feats)
        data = Data(x=x, edge_index=edge_index)
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.long)
        return data

    # --- Paso 3: Crear datasets ---
    dataset_train = [
        row_to_data(X_train.loc[idx], edge_index, y_train.loc[idx])
        for idx in X_train.index
    ]

    dataset_test = [
        row_to_data(X_test.loc[idx], edge_index, y_test.loc[idx])
        for idx in X_test.index
    ]

    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=16)

    # --- Paso 4: Definir modelo ---
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_feats, 16)
            self.conv2 = GCNConv(16, 32)
            self.fc = torch.nn.Linear(32, 2)  

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)  
            return F.log_softmax(self.fc(x), dim=1)
        
    # Alternativa: Definir un modelo GAT
        
    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(num_feats, 16, heads=4, concat=True)  
            self.conv2 = GATConv(16 * 4, 32)  
            self.fc = torch.nn.Linear(32, 2)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return F.log_softmax(self.fc(x), dim=1)

    # --- Paso 5: Entrenamiento ---
    if model == 'GAT':

        model = GAT()
    else:

        model = GCN()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch in loader_train:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # if epoch % 10 == 0:
            # print(f"Epoch {epoch}, Loss: {total_loss / len(loader_train):.4f}")

    # --- Paso 6: Evaluación ---
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader_test:
            out = model(batch)
            probs = torch.exp(out)
            preds = probs.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())  # probability of class 1
            all_labels.extend(batch.y.tolist())

    accuracy = correct / total

    return accuracy, all_preds, all_probs



def graph_kernel(G_base, X_train, X_test, y_train, y_test):

    import networkx as nx
    from grakel import Graph
    from sklearn.svm import SVC
    from grakel.kernels import WeisfeilerLehman
    
    def construir_grafo_muestra(G_base, expr_muestra, medianas):
        G = G_base.copy()
        valores_expr = expr_muestra.values
        # assert len(G.nodes) == len(valores_expr)
        for i, nodo in enumerate(G.nodes()):
            valor = valores_expr[i]
            umbral = medianas.iloc[i]
            G.nodes[nodo]['label'] = int(valor > umbral)
        return G


    def convertir_a_grakel(G_nx):
        """
        Convierte un grafo de networkx al formato GraKeL.
        """
        etiquetas = nx.get_node_attributes(G_nx, 'label')
        edges = list(G_nx.edges())
        node_labels = {n: etiquetas[n] for n in G_nx.nodes()}

        return Graph(edges, node_labels=node_labels)
    

    medianas = X_train.median(axis=0)

    # Construye grafos por muestra
    grafos_train = [construir_grafo_muestra(G_base, expr, medianas) for _, expr in X_train.iterrows()]
    grafos_test  = [construir_grafo_muestra(G_base, expr, medianas) for _, expr in X_test.iterrows()]

    # Convierte a formato GraKeL
    gra_k_train = [convertir_a_grakel(G) for G in grafos_train]
    gra_k_test  = [convertir_a_grakel(G) for G in grafos_test]

    # Kernel que usa atributos continuos
    gk = WeisfeilerLehman(n_iter=3)

    # Calculamos matrices kernel
    K_train = gk.fit_transform(gra_k_train)
    K_test = gk.transform(gra_k_test)

    # Clasificador con kernel precomputado
    clf = SVC(kernel='precomputed', C=1, probability=True)
    clf.fit(K_train, y_train)

    # Predicción y accuracy
    y_pred = clf.predict(K_test)

    y_proba = clf.predict_proba(K_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    return acc, y_pred, y_proba


bbdd = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'skcm', 'sarc', 'stad', 'ucec']

for bd in bbdd:
    data = pd.read_csv('data/main_experiments/data_plos_' + bd + '_lognorm.csv', sep=',', index_col=0)
    data.rename(columns={data.columns[-1]: 'classvalues'}, inplace=True)

    X = data.drop('classvalues', axis=1)
    y = data['classvalues']

    mrnc = rnacorex.MRNC(precision = 20)
    mrnc.initialize_model(X, y)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)

    metrics_gnn = []                                                                                                                              
    metrics_gker = []
    metrics_rnacorex = []
    metrics_rf = []
    metrics_svm = []
    metrics_gb = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(mrnc.X_, mrnc.y_)):

        print(f'Fold: {fold}')
        X_train, y_train = mrnc.X_.iloc[train_idx], mrnc.y_.iloc[train_idx]
        X_test, y_test = mrnc.X_.iloc[test_idx], mrnc.y_.iloc[test_idx]

        mrnc.compute_functional(X_train, y_train)
        mrnc.rank()


        for k in tqdm(range(2, 200)):

            mrnc.n_con = k
            mrnc.fit_only(new_sets=True)

            y_pred_rna = mrnc.predict(X_test)
            y_prob_rna = mrnc.predict_proba(X_test)[:,1]

            acc_rna = (y_pred_rna == y_test).mean()
            auc_rna = roc_auc_score(y_test, y_prob_rna)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rna).ravel()
            sens_rna = tp / (tp + fn)
            spec_rna = tn / (tn + fp)
            metrics_rnacorex.append({
                "fold": fold, "k": k,
                "accuracy": acc_rna, "auc": auc_rna,
                "sensitivity": sens_rna, "specificity": spec_rna
            })

            G_base = mrnc.clgc_['G']
            X_train_temp = X_train[mrnc.clgc_['X'].columns]
            X_test_temp = X_test[mrnc.clgc_['X'].columns]

            acc_gnn, pred_gnn, prob_gnn = GNN(G_base, X_train_temp, X_test_temp, y_train, y_test, model='GAT')
            auc_gnn = roc_auc_score(y_test, prob_gnn)
            tn, fp, fn, tp = confusion_matrix(y_test, pred_gnn).ravel()
            sens_gnn = tp / (tp + fn)
            spec_gnn = tn / (tn + fp)
            metrics_gnn.append({
                "fold": fold, "k": k,
                "accuracy": acc_gnn, "auc": auc_gnn,
                "sensitivity": sens_gnn, "specificity": spec_gnn
            })

            acc_gker, pred_gker, prob_gker = graph_kernel(G_base, X_train_temp, X_test_temp, y_train, y_test)
            auc_gker = roc_auc_score(y_test, prob_gker)
            tn, fp, fn, tp = confusion_matrix(y_test, pred_gker).ravel()
            sens_gker = tp / (tp + fn)
            spec_gker = tn / (tn + fp)
            metrics_gker.append({
                "fold": fold, "k": k,
                "accuracy": acc_gker, "auc": auc_gker,
                "sensitivity": sens_gker, "specificity": spec_gker
            })

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_temp, y_train)
            pred_rf = rf.predict(X_test_temp)
            prob_rf = rf.predict_proba(X_test_temp)[:, 1]
            auc_rf = roc_auc_score(y_test, prob_rf)
            acc_rf = accuracy_score(y_test, pred_rf)
            tn, fp, fn, tp = confusion_matrix(y_test, pred_rf).ravel()
            sens_rf = tp / (tp + fn)
            spec_rf = tn / (tn + fp)
            metrics_rf.append({"fold": fold, "k": k, "accuracy": acc_rf, "auc": auc_rf,
                "sensitivity": sens_rf, "specificity": spec_rf})
            
            svm = SVC(probability=True, kernel='rbf', random_state=42)
            svm.fit(X_train_temp, y_train)
            pred_svm = svm.predict(X_test_temp)
            prob_svm = svm.predict_proba(X_test_temp)[:, 1]
            auc_svm = roc_auc_score(y_test, prob_svm)
            acc_svm = accuracy_score(y_test, pred_svm)
            tn, fp, fn, tp = confusion_matrix(y_test, pred_svm).ravel()
            sens_svm = tp / (tp + fn)
            spec_svm = tn / (tn + fp)
            metrics_svm.append({"fold": fold, "k": k, "accuracy": acc_svm, "auc": auc_svm,
                "sensitivity": sens_svm, "specificity": spec_svm})
            
            gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, random_state=42)
            gb.fit(X_train_temp, y_train)
            pred_gb = gb.predict(X_test_temp)
            prob_gb = gb.predict_proba(X_test_temp)[:, 1]
            auc_gb = roc_auc_score(y_test, prob_gb)
            acc_gb = accuracy_score(y_test, pred_gb)
            tn, fp, fn, tp = confusion_matrix(y_test, pred_gb).ravel()
            sens_gb = tp / (tp + fn)
            spec_gb = tn / (tn + fp)
            metrics_gb.append({"fold": fold, "k": k, "accuracy": acc_gb, "auc": auc_gb,
                "sensitivity": sens_gb, "specificity": spec_gb})

    # Convert results to DataFrames

    df_rna = pd.DataFrame(metrics_rnacorex)
    df_gnn = pd.DataFrame(metrics_gnn)
    df_gker = pd.DataFrame(metrics_gker)
    df_rf = pd.DataFrame(metrics_rf)
    df_svm = pd.DataFrame(metrics_svm)
    df_gb = pd.DataFrame(metrics_gb)

    # Save all sheets to Excel
    # with pd.ExcelWriter(f'Results/metrics_def_{bd}.xlsx') as writer:
    #     df_rna.to_excel(writer, sheet_name='rnacorex', index=False)
    #     df_gnn.to_excel(writer, sheet_name='gnn', index=False)
    #     df_gker.to_excel(writer, sheet_name='gker', index=False)
    #     df_rf.to_excel(writer, sheet_name='rf', index=False)
    #     df_svm.to_excel(writer, sheet_name='svm', index=False)
    #     df_gb.to_excel(writer, sheet_name='gb', index=False)

elapsed = time.time() - start_time
print(f"Script completed in {elapsed:.2f} seconds.")
