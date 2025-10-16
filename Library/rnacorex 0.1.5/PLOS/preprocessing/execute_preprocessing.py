from preprocessing import preprocessing_rnacorex

bbdds = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

for bbdd in bbdds:
    print(f'Processing {bbdd}...')
    preprocessing_rnacorex(bbdd)