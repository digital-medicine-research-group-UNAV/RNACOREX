import pandas as pd

bbdds = ['brca', 'coad', 'hnsc', 'kirc', 'laml', 'lgg', 'lihc', 'luad', 'lusc', 'sarc', 'skcm', 'stad', 'ucec']

for bd in bbdds:

    file_path = '../results/metrics_def_'+bd+'_lognorm.xlsx'
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    summary_sheets = {}

    for sheet_name, df in all_sheets.items():
        df_avg = df.groupby("k")[["accuracy", "auc", "sensitivity", "specificity"]].mean().reset_index()

        # Encontrar el k con mayor accuracy media
        best_idx = df_avg["accuracy"].idxmax()
        best_k = df_avg.loc[best_idx, "k"]
        best_acc = df_avg.loc[best_idx, "accuracy"]

        final_summary = pd.DataFrame({
            "metric": ["accuracy", "auc", "sensitivity", "specificity"],
            "mean": df_avg[["accuracy", "auc", "sensitivity", "specificity"]].mean().values,
            "max": df_avg[["accuracy", "auc", "sensitivity", "specificity"]].max().values,
            "std": df_avg[["accuracy", "auc", "sensitivity", "specificity"]].std().values
        })

        # Agregar fila con el mejor k
        best_row = pd.DataFrame({
            "metric": ["best_k_for_accuracy"],
            "mean": [best_k],
            "max": [best_acc],
            "std": [None]
        })

        final_summary = pd.concat([final_summary, best_row], ignore_index=True)

        summary_sheets[sheet_name] = final_summary

    with pd.ExcelWriter('../results/table_results_'+bd+'.xlsx') as writer:
        for sheet_name, df_summary in summary_sheets.items():
            df_summary.to_excel(writer, sheet_name=sheet_name, index=False)