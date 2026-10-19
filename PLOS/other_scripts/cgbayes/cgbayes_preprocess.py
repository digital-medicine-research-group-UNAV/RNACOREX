import pandas as pd
from sklearn.model_selection import train_test_split

# Preprocess data to get train and test sets for CGBayesNets benchmarking.

bbdds = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

for bd in bbdds:

    data = pd.read_csv('../../data/main_experiments/data_plos_'+bd+'_lognorm.csv', index_col=0)

    train_df, test_df = train_test_split(
        data,
        train_size=50,
        test_size=20,
        stratify=data['Class'],
        random_state=42
    )

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train class distribution:\n", train_df['Class'].value_counts())
    print("Test class distribution:\n", test_df['Class'].value_counts())

    train_df.to_csv('../../data/cgbayesnets/data_train_'+bd+'_lognorm.csv', index=False)
    test_df.to_csv('../../data/cgbayesnets/data_test_'+bd+'_lognorm.csv', index=False)
