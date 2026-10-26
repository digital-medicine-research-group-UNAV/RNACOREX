import pandas as pd
import os
from tqdm import tqdm



def eliminate_downexpressed(mrna, lncrna, mirna, perc_mrna = 0.75, perc_mirna = 0.75, perc_lncrna = 0.75, count_mrna = 5, count_mirna = 1, count_lncrna = 3):
    
    # Porcentaje de elementos menores que count (<5) es menor que 100% - 75% = 25%.

    condition_mrna = mrna.apply(lambda col: (col <= count_mrna).sum() / len(col) < (1-perc_mrna))

    condition_lncrna = lncrna.apply(lambda col: (col <= count_lncrna).sum() / len(col) < (1-perc_lncrna))

    condition_mirna = mirna.apply(lambda col: (col <= count_mirna).sum() / len(col) < (1-perc_mirna))

    mirna_filt = mirna.loc[:, condition_mirna]

    mrna_filt = mrna.loc[:, condition_mrna]
    
    lncrna_filt = lncrna.loc[:, condition_lncrna]

    filtered_dataset = pd.concat([mrna, lncrna, mirna], axis=1)

    return {"filtered_dataset": filtered_dataset, "mrna": mrna_filt, "lncrna": lncrna_filt, "mirna": mirna_filt}



def deg(counts_data, class_data, class_A, class_B, alpha):

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    metadata_df = pd.DataFrame({
        "sample": counts_data.index.values,
        "Class": class_data
    })

    metadata_df.set_index("sample", inplace=True)

    counts_data = counts_data.astype(int)
    metadata_df = metadata_df.astype(str)

    inference = DefaultInference(n_cpus=1)

    dds = DeseqDataSet(
        counts=counts_data,
        metadata=metadata_df,
        design_factors=class_data.name,
        refit_cooks=False,
        inference=inference,
    )

    dds.fit_LFC()

    ds = DeseqStats(
        dds,
        contrast=[class_data.name, class_A, class_B],
        alpha=0.1,
        cooks_filter=False,
        independent_filter=False,
        inference = inference
    )

    ds.run_wald_test()

    ds.summary()

    deg_elements = counts_data[ds.padj[ds.padj < alpha].index.values]

    return deg_elements



def filter_db(X, mrna, mirna, lncrna, engine):

    # Load interactions
    if engine is None:
        module_path = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(module_path, '../..', 'Engine', 'interactions.parquet')
        engine = pd.read_parquet(engine_path, engine="pyarrow")
        print('USING DEFAULT ENGINE')
    else:
        print('USING PERSONALIZED ENGINE')

    if mrna == False:
        engine = engine[(engine['Type1'] != 'mRNA') & (engine['Type2'] != 'mRNA')]

    if mirna == False:
        engine = engine[(engine['Type1'] != 'miRNA') & (engine['Type2'] != 'miRNA')]

    if lncrna == False:
        engine = engine[(engine['Type1'] != 'lncRNA') & (engine['Type2'] != 'lncRNA')]
    
    # Filter the input database
    valid_values = list(X.columns)
    filtered_engine = engine[(engine['Gene1'].isin(valid_values)) & (engine['Gene2'].isin(valid_values))]
    
    valid_columns = set(filtered_engine['Gene1']).union(set(filtered_engine['Gene2']))
    valid_columns = list(filter(lambda col: col in valid_columns, X.columns))
    X = X[valid_columns]
    
    labels = list(X.columns)

    gene1_series = filtered_engine[['Gene1', 'Type1', 'Gene_name1']].drop_duplicates('Gene1').set_index('Gene1')
    gene2_series = filtered_engine[['Gene2', 'Type2', 'Gene_name2']].drop_duplicates('Gene2').set_index('Gene2')
    
    types = []
    gene_names = []
    
    for label in labels:
        if label in gene1_series.index:
            type_value = gene1_series.loc[label, 'Type1']
            gene_name = gene1_series.loc[label, 'Gene_name1']
        elif label in gene2_series.index:
            type_value = gene2_series.loc[label, 'Type2']
            gene_name = gene2_series.loc[label, 'Gene_name2']
        else:
            continue
        
        types.append(type_value)
        gene_names.append(gene_name)
    
    element_info = {
        'labels': labels,
        'names': gene_names,
        'types': types
    }

    return X, filtered_engine, element_info