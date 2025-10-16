import pandas as pd
import os
from gtfparse import read_gtf
import os
import sys
import numpy as np
from scipy.stats import zscore


def preprocessing_rnacorex(bbdd):

    # Se eliminan aquellos mRNA en los que hay más de un 25% de muestras con menos de 5 counts y aquellos miRNA en los que hay más de un 25% de muestras con menos de 1 count.

    def eliminate_downexpressed(mrna, mirna, perc_mrna = 0.75, perc_mirna = 0.75, count_mrna = 5, count_mirna = 1):
    
        # Porcentaje de elementos menores que count (<5) es menor que 100% - 75% = 25%.

        condition_mrna = mrna.apply(lambda col: (col <= count_mrna).sum() / len(col) < (1-perc_mrna))

        condition_mirna = mirna.apply(lambda col: (col <= count_mirna).sum() / len(col) < (1-perc_mirna))

        mirna_filt = mirna.loc[:, condition_mirna]

        mrna_filt = mrna.loc[:, condition_mrna]

        filtered_dataset = pd.concat([mrna, mirna], axis=1)

        return {"filtered_dataset": filtered_dataset, "mrna": mrna_filt, "mirna": mirna_filt}


    def deg(counts_data, class_data, class_A, class_B, alpha):

        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.preprocessing import deseq2_norm

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
        normalized_counts, _ = deseq2_norm(counts_data)

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

        results_df = ds.results_df.copy()
        results_df = results_df.sort_values("padj", ascending=True)

        deg_results = results_df[results_df["padj"] < alpha]

        deg_elements = counts_data[ds.padj[ds.padj < alpha].index.values]
        normalized_counts = normalized_counts[ds.padj[ds.padj < alpha].index.values]

        return deg_elements, deg_results, normalized_counts

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

    data_mrna_path = os.path.join('TCGA RAW/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.star_counts.tsv')
    data_mirna_path = os.path.join('TCGA RAW/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.mirna.tsv')
    data_survival_path = os.path.join('TCGA RAW/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.survival.tsv')

    data_mrna = pd.read_csv(data_mrna_path, sep = '\t')
    data_mirna = pd.read_csv(data_mirna_path, sep = '\t')
    data_survival = pd.read_csv(data_survival_path, sep = '\t')

    gtf_path = os.path.join('../../engines/gencode.v47.basic.annotation.gtf')

    gencode = read_gtf(gtf_path).to_pandas()

    # ELIMINAR MUESTRAS PERITUMORALES

    last_part = data_survival['sample'].str.split('-').str[-1]
    data_survival = data_survival[last_part.str.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '09', '00'))]
    data_survival.index = data_survival['sample']
    data_survival = data_survival.drop(['sample'], axis=1)

    # PREPROCESADO DE MRNA

    data_mrna = data_mrna.T
    data_mrna.columns = data_mrna.iloc[0]
    data_mrna = data_mrna.iloc[1:,:]

    # PREPROCESADO DE MIRNA

    data_mirna = data_mirna.T
    data_mirna.columns = data_mirna.iloc[0]
    data_mirna = data_mirna.iloc[1:,:]

    # Pasar a Counts

    data_mrna_transformed = 2**data_mrna - 1

    data_mrna_transformed = data_mrna_transformed.astype(float)
    data_mirna = data_mirna.astype(float)

    # Unir las bases de datos para tener los mismos pacientes en todas.

    df_mrna_mirna = data_mrna_transformed.join(data_mirna, how='inner')
    df_merged = df_mrna_mirna.join(data_survival, how='inner')

    # Balancear las clases 
    # (OS = 0, censurado, vivo)
    # (OS = 1, no censurado, muerto)
    # (Class = 0, short survival)
    # (Class = 1, long survival)

    data_survival = df_merged[['OS', 'OS.time']]

    data_survival['Class'] = -1

    p75_no_censurados = data_survival[data_survival['OS'] == 1]['OS.time'].quantile(0.75)

    data_survival.loc[(data_survival['OS'] == 1) & (data_survival['OS.time'] <= p75_no_censurados), 'Class'] = 0
    data_survival.loc[(data_survival['OS'] == 1) & (data_survival['OS.time'] > p75_no_censurados), 'Class'] = 1

    n_class_0 = (data_survival['Class'] == 0).sum()
    n_class_1 = (data_survival['Class'] == 1).sum()

    # Filtrar los censurados (OS = 0) con OS.time > P75 de los no censurados
    data_censored = data_survival[(data_survival['OS'] == 0) & (data_survival['OS.time'] > p75_no_censurados)]

    # Ordenar por OS.time descendente para seleccionar los que tienen el tiempo más alto
    data_censored_sorted = data_censored.sort_values(by='OS.time', ascending=False)

    # Seleccionar la cantidad necesaria de censurados para igualar las clases
    n_needed = n_class_0 - n_class_1  # Cuántos faltan en Class = 1

    if n_needed > 0:
        data_censored_selected = data_censored_sorted.head(n_needed)  # Tomar los primeros (más altos)
        data_survival.loc[data_censored_selected.index, 'Class'] = 1  # Asignarles Class = 1

    data_complete = pd.concat([df_merged, data_survival], axis = 1)
    data_complete = data_complete[data_complete['Class'] != -1]

    # SEPARAR EN MIRNA, MRNA Y SURVIVAL

    data_mrna = data_complete.loc[:, data_complete.columns.str.startswith('ENSG')]
    data_mirna = data_complete.loc[:, data_complete.columns.str.startswith('hsa')]
    classvalues = data_complete['Class']

    filtered_data = eliminate_downexpressed(data_mrna, data_mirna)

    # DGE (USING PYDESEQ2)

    deg_mrna, deg_results, normalized_counts = deg(filtered_data["mrna"], classvalues, '0', '1', 0.05)

    deg_data = pd.concat([deg_mrna, filtered_data["mirna"]], axis = 1)

    normalized_pydeseq = pd.concat([normalized_counts, filtered_data["mirna"]], axis = 1)

    deg_data = deg_data.apply(lambda x: np.log2(x + 1))

    # deg_data = deg_data.apply(zscore) # For PLOS results apply zscore

    data_def = pd.concat([deg_data, classvalues], axis=1)

    normalized_def = pd.concat([normalized_pydeseq, classvalues], axis=1)

    data_def.columns = data_def.columns.str.split('.').str[0]

    normalized_pydeseq.columns = normalized_pydeseq.columns.str.split('.').str[0]

    data_def.to_csv('../data/data_plos_'+bbdd+'_lognorm.csv') # Log-normalized data.

    # deg_results.to_csv('../results/deg_results_'+bbdd+'_pydeseq.csv') # DEG results from pydeseq2.

    # normalized_pydeseq.to_csv('../data/data_plos_'+bbdd+'_pydeseq.csv') # Normalized data from pydeseq2.
