from data_preprocessing_general import preprocesado_bbdd
from tqdm import tqdm 

notebook_names = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'ov', 'sarc', 'skcm', 'stad', 'ucec']

for notebook in tqdm(notebook_names):
    
    print('PREPROCESSING ', notebook.upper())

    if notebook == 'skcm' or notebook == 'ov' or notebook =='laml' or notebook == 'lgg':
        preprocesado_bbdd(notebook, 0.5, 0.7, 0.05, only_noncensored = True)

    else:
        preprocesado_bbdd(notebook, 0.5, 0.7, 0.05)

    