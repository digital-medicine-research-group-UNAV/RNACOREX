import sys
import os
import pandas as pd
from gtfparse import read_gtf

miRNetClassifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'miRNetClassifier'))
sys.path.append(miRNetClassifier_path)

import AuxiliaryFunctions

def preprocessing_1(bbdd, umbral_na, umbral_variabilidad, only_noncensored = False): 

    data_mrna = pd.read_csv('TCGA Raw Data/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.htseq_counts.tsv', sep = '\t')
    data_mirna = pd.read_csv('TCGA Raw Data/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.mirna.tsv', sep = '\t')
    data_survival = pd.read_csv('TCGA Raw Data/'+bbdd.upper()+'/TCGA-'+bbdd.upper()+'.survival.tsv', sep = '\t')

    gtf_path = os.path.join(miRNetClassifier_path, 'gtf', 'gencode.v45.chr_patch_hapl_scaff.basic.annotation.gtf')

    gtf = read_gtf(gtf_path)
    # gtf = read_gtf('miRNetClassifier/gtf/gencode.v45.chr_patch_hapl_scaff.basic.annotation.gtf.gz')

    # ELIMINAR MUESTRAS PERITUMORALES

    last_part = data_survival['sample'].str.split('-').str[-1]
    data_survival = data_survival[last_part.str.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '09', '00'))]
    data_survival.index = data_survival['sample']
    data_survival = data_survival.drop(['sample'], axis=1)

    # PREPROCESADO DE MRNA

    data_mrna = data_mrna.T
    data_mrna.columns = data_mrna.iloc[0]
    data_mrna = data_mrna.iloc[1:,:]

    # CAMBIAR NOMBRE DE GENES A FORMATO ENSG

    data_mrna.columns = AuxiliaryFunctions.rename_all(data_mrna.columns, gtf)

    # ELIMINAR TRANSCRITOS NO MRNA

    data_mrna.drop(columns=[col for col in data_mrna.columns if not col.startswith('ENSG0')], inplace=True)
    data_mirna = data_mirna.T
    data_mirna.columns = data_mirna.iloc[0]
    data_mirna = data_mirna.iloc[1:,:]

    # UNIR DATAFRAMES DE MRNA, MIRNA Y SURVIVAL PARA TENER SÓLAMENTE MUESTRAS TUMORALES DE PACIENTES CON DATOS DE AMBAS COSAS

    df_mrna_mirna = data_mrna.join(data_mirna, how='inner')
    df_merged = df_mrna_mirna.join(data_survival, how='inner')

    # ELIMINAR GENES Y MICROS CON MÁS DE UN 50% DE VALORES PERDIDOS, RESTO IMPUTAR CON LA MEDIANA

    filtered_df = df_merged.filter(regex='^(ENSG|hsa)')
    umbral = len(filtered_df) * umbral_na
    filtered_df = filtered_df.dropna(thresh=umbral, axis=1)
    filtered_df.fillna(filtered_df.median())

    # ELIMINAR GENES Y MICROS CON VALORES DE EXPRESIÓN REPETIDOS EN MÁS DE UN 70% DE LAS MUESTRAS

    repeated_percentage = filtered_df.apply(lambda col: col.value_counts(normalize=True).max())
    threshold = umbral_variabilidad
    columns_to_drop = repeated_percentage[repeated_percentage > threshold].index
    filtered_df.drop(columns=columns_to_drop, inplace=True)

    if only_noncensored == True:

        df_joined = df_merged[['OS', 'OS.time']].join(filtered_df, how='inner')
        median_value = df_joined[df_joined['OS'] == 1]['OS.time'].quantile(0.5)
        df_joined2 = df_joined[df_joined['OS'] == 1]
        df_joined2['OS_dic'] = (pd.to_numeric(df_joined2['OS.time']) > median_value).astype(int)
        df_joined2 = df_joined2.drop(columns = ['OS', 'OS.time'], axis = 1)
        df_joined2.rename(columns={'OS_dic': 'classvalues'}, inplace=True)
        df_joined2 = df_joined2.apply(pd.to_numeric, errors='coerce')

    else:

        # DEFINIR LAS CLASES <P75 OS=1, >P75 + OS=0

        df_joined = df_merged[['OS', 'OS.time']].join(filtered_df, how='inner')
        median_value = df_joined[df_joined['OS'] == 1]['OS.time'].quantile(0.75)
        data_censored = df_joined[df_joined['OS'] == 0]
        data_censored_1 = data_censored[data_censored['OS.time'] > median_value].iloc[-round((1/2)*len(df_joined[df_joined['OS'] == 1])):, :]
        data_no_censored = df_joined[df_joined['OS'] == 1]

        # GENERAR EL DATAFRAME DEFINITIVO

        df_joined2 = pd.concat([data_no_censored, data_censored_1], ignore_index=True)
        df_joined2['OS_dic'] = (pd.to_numeric(df_joined2['OS.time']) > median_value).astype(int)
        df_joined2 = df_joined2.drop(columns = ['OS', 'OS.time'], axis = 1)
        df_joined2.rename(columns={'OS_dic': 'classvalues'}, inplace=True)
        df_joined2 = df_joined2.apply(pd.to_numeric, errors='coerce')

    # EXPRESIÓN DIFERENCIAL DE GENES

    data_deg = df_joined2.filter(regex='^ENSG')
    data_mirna = df_joined2.filter(regex='^hsa')

    data_deg['classvalues'] = df_joined2['classvalues']
    data_deg.to_csv('DEG R/data_deg_'+bbdd+'.csv')
    df_joined2.to_csv('DEG R/data_complete_'+bbdd+'.csv')

    # At this point differential gene expression is done in R with edgeR. Differentially expressed genes are saved in deg_genes_bbdd*.

def preprocessing_2(bbdd):

    genes_deg = pd.read_csv('DEG R/deg_genes_'+bbdd+'.csv')
    data_deg = pd.read_csv('DEG R/data_deg_'+bbdd+'.csv')
    df_joined2 = pd.read_csv('DEG R/data_complete_'+bbdd+'.csv')
    columns_to_ignore = [col for col in data_deg.columns if col not in list(genes_deg.iloc[:,0])]
    df_clean = df_joined2.drop(columns=columns_to_ignore)
    df_clean['classvalues'] = df_joined2['classvalues'] 

    # ELIMINAR LA PARTE FINAL DEL CÓDIGO ENSG (DESPUÉS DEL PUNTO)

    df_clean.columns = df_clean.columns.str.split('.').str[0]

    # GUARDAR ARCHIVO

    df_clean.to_csv('Clean Data/SampleData'+bbdd.upper()+'.csv')