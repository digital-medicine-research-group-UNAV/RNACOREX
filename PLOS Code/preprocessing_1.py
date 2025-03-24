from data_preprocessing import preprocessing_1
from tqdm import tqdm
    

notebook_names = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'ov', 'sarc', 'skcm', 'stad', 'ucec']

for notebook in tqdm(notebook_names):
    
    print('PREPROCESSING ', notebook.upper())

    if notebook == 'skcm' or notebook == 'ov' or notebook =='laml' or notebook == 'lgg':
        preprocessing_1(notebook, 0.5, 0.7, only_noncensored = True)

    else:
        preprocessing_1(notebook, 0.5, 0.7)