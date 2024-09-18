from data_preprocessing_general import preprocessing_2
from tqdm import tqdm

notebook_names = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'ov', 'sarc', 'skcm', 'stad', 'ucec']

for notebook in tqdm(notebook_names):

    preprocessing_2(notebook)