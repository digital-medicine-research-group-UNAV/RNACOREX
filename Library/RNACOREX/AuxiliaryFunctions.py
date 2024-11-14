import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from gtfparse import read_gtf


def rename_all(gene_list, gtf):

    """
    Using gtfparse converts a list of gene names in Hugo Symbols to Ensembl nomenclature.
    
    Args:
            gene_list (list): list of Hugo gene names.
            gtf (gtf): gtf file.
    
    Returns:
            renamed_list (list): list of renamed genes in Ensembl nomenclature.
            
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    conversor_path = os.path.join(current_dir, 'gtf', 'conversor2.txt')
    extra = pd.read_csv(conversor_path, sep='\s+', header=None)
    extra.columns = ['gene_name', 'gene_id']
    new_df = pd.DataFrame()
    new_df['gene_id'] = gtf['gene_id']
    new_df['gene_name'] = gtf['gene_name']
    new_df['gene_id'] = [s.split('.')[0] for s in new_df['gene_id']]
    df_extra = pd.concat([new_df, extra], ignore_index=True)
    df_extra.drop_duplicates(keep = 'first', inplace = True)
    result_dict = dict(zip(df_extra['gene_name'], df_extra['gene_id']))
    mapped_list = [result_dict.get(item, item) for item in gene_list]
    return mapped_list

def ens_to_gen(ens_list, gtf):

    """
    Converts a list of ENSG gene names to Hugo Symbol.
    
    Args:
            ens_list (list): list of ENSG gene names.
            gtf (gtf): gtf file.
    
    Returns:
    
            renamed_list (list): list of renamed genes in Hugo Symbols.    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    conversor_path = os.path.join(current_dir, 'gtf', 'conversor2.txt')
    extra = pd.read_csv(conversor_path, sep='\s+', header=None)
    extra.columns = ['gene_name', 'gene_id']
    new_df = pd.DataFrame()
    new_df['gene_id'] = gtf['gene_id']
    new_df['gene_name'] = gtf['gene_name']
    new_df['gene_id'] = [s.split('.')[0] for s in new_df['gene_id']]
    df_extra = pd.concat([new_df, extra], ignore_index=True)
    df_extra = df_extra[~df_extra['gene_id'].str.startswith('#')]
    filter_mask = df_extra['gene_id'].isin(ens_list)
    filtered_df = df_extra[filter_mask]
    first_occurrences = filtered_df.drop_duplicates(subset='gene_name', keep='first')
    column_mapping = first_occurrences.set_index('gene_id')['gene_name'].to_dict()
    renamed_list = [column_mapping.get(item, item) for item in ens_list]

    return renamed_list

def rename_genes(gene_list, gtf):

    """
    Using gtfparse converts a list of gene names in Hugo Symbols to Ensembl nomenclature..
    
    Args:
            gene_list (list): list of Hugo gene names.
            gtf (gtf): gtf file.
    
    Returns:
            renamed_list (list): list of renamed genes in Ensembl nomenclature.
            
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    conversor_path = os.path.join(current_dir, 'gtf', 'conversor2.txt')
    extra = pd.read_csv(conversor_path, sep='\s+', header=None)
    extra.columns = ['gene_name', 'gene_id']

    new_df = pd.DataFrame()
    new_df['gene_id'] = gtf['gene_id']
    new_df['gene_name'] = gtf['gene_name']
    new_df['gene_id'] = [s.split('.')[0] for s in new_df['gene_id']]
    df_extra = pd.concat([new_df, extra], ignore_index=True)
    df_extra = df_extra[~df_extra['gene_id'].str.startswith('#')]

    filter_mask = df_extra['gene_name'].isin(gene_list)
    filtered_df = df_extra[filter_mask]
    first_occurrences = filtered_df.drop_duplicates(subset='gene_id', keep='first')
    column_mapping = first_occurrences.set_index('gene_name')['gene_id'].to_dict()
    renamed_list = [column_mapping.get(item, item) for item in gene_list]

    return renamed_list



def benjamini_hochberg(genes, test = 'mann_whitney'):

    """
    
    Performs a Gene Differential Expression analysis with Benjamini-Hochberg correction on a list of genes.
    
    Args:
            genes (pd.DataFrame): DataFrame with gene expression values.
            test (str): statistical test to be used. Default is Mann-Whitney.
            
    Returns:
    
            pvalues_corrected_g (pd.DataFrame): DataFrame with corrected p-values.
            
    """
    group0_g = genes[genes['classvalues'] == 0]
    group1_g = genes[genes['classvalues'] == 1]

    group0_g = group0_g.drop('classvalues', axis = 1)
    group1_g = group1_g.drop('classvalues', axis = 1)

    p_values_g = []

    print('EXTRAYENDO GENES SIGNIFICATIVOS')
    for rna in tqdm(group0_g.columns):
        if test == 'mann_whitney':
            t_statistic, p_value = mannwhitneyu(group0_g[rna], group1_g[rna])
            p_values_g.append(p_value)
        if test == 't_test':
            t_statistic, p_value = ttest_ind(group0_g[rna], group1_g[rna])    
            p_values_g.append(p_value)

    pvalues_g = pd.DataFrame()
    pvalues_g['pvalue'] = p_values_g
    pvalues_g.index = genes.drop(['classvalues'], axis=1).columns
    pvalues_g = pvalues_g.sort_values(by='pvalue')

    corrected_p_values_g = multipletests(p_values_g, method='fdr_bh')[1]

    pvalues_corrected_g = pd.DataFrame()
    pvalues_corrected_g['pvalue'] = corrected_p_values_g
    pvalues_corrected_g.index = genes.drop(['classvalues'], axis=1).columns

    pvalues_corrected_g = pvalues_corrected_g.sort_values(by='pvalue')
    
    return pvalues_corrected_g, pvalues_g


def get_genes(pvalues_g, alpha = 0.05, number_genes = 0):

    """
    
    Extracts the significant genes from a list of p-values.
    
    Args:
    
            pvalues_g (pd.DataFrame): DataFrame with p-values.
            alpha (int): significance level.
            number_genes (int): number of genes to be extracted. Default is 0. If 0, all significant genes are extracted.
    
    Returns:
    
            genes (list): list of significant genes.
    
    """

    if number_genes == 0:
        genes = pvalues_g[pvalues_g['pvalue'] < alpha].index

    else:
        genes = pvalues_g.index[:number_genes]

    return genes
