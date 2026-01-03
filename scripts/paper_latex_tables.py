import glob
import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

NUM_THRESHOLDS = 7
LANGUAGES_PER_DATASET = {
    "panx": 173,
    "masakhaner": 20,
    "multinerd": 10,
    "multiconer_v1": 13,
    "multiconer_v2": 13,
    "dynamicner": 8,
    "uner": 10,
    "panx_translated": 173,
    "masakhaner_translated": 20,
    "multinerd_translated": 10,
    "multiconer_v1_translated": 13,
    "multiconer_v2_translated": 13,
    "dynamicner_translated": 8,
    "uner_translated": 10,
}

EXPERIMENTS = [
    "bi_rembert_finerweb",
    "bi_rembert_finerweb-translated",
    "bi_rembert_euroglinerx",
    "bi_rembert_pilener",
    "bi_mdeberta_finerweb",
    "bi_mdeberta_finerweb-translated",
    "bi_mdeberta_euroglinerx",
    "bi_mdeberta_pilener",
    "bi_xlmr_finerweb",
    "bi_xlmr_finerweb-translated",
    "bi_xlmr_euroglinerx",
    "bi_xlmr_pilener",
    "bi_mmbert_finerweb",
    "bi_mmbert_finerweb-translated",
    "bi_mmbert_euroglinerx",
    "bi_mmbert_pilener",
    "bi_mt5_finerweb",
    "bi_mt5_finerweb-translated",
    "bi_mt5_euroglinerx",
    "bi_mt5_pilener",
    "ce_rembert_finerweb",
    "ce_rembert_finerweb-translated",
    "ce_rembert_euroglinerx",
    "ce_rembert_pilener",
    "ce_mdeberta_finerweb",
    "ce_mdeberta_finerweb-translated",
    "ce_mdeberta_euroglinerx",
    "ce_mdeberta_pilener",
    "ce_xlmr_finerweb",
    "ce_xlmr_finerweb-translated",
    "ce_xlmr_euroglinerx",
    "ce_xlmr_pilener",
    "ce_mmbert_finerweb",
    "ce_mmbert_finerweb-translated",
    "ce_mmbert_euroglinerx",
    "ce_mmbert_pilener",
    "ce_mt5_finerweb",
    "ce_mt5_finerweb-translated",
    "ce_mt5_euroglinerx",
    "ce_mt5_pilener",
]

def plot_thresholds(df):
    # first, we need to figure out scores for each threshold
    # exclude translated evaluation dirs
    df_thresholds = df[~df['evaluation_dir'].str.contains('translated')]
    df_thresholds = df_thresholds[df_thresholds['evaluation_dir'].isin(['panx', 'masakhaner', 'multinerd', 'dynamicner', 'uner'])]
    # Create an 'architecture' column by extracting 'ce' or 'bi' from the experiment name
    df_thresholds['architecture'] = df_thresholds['experiment'].str.extract(r'^(ce|bi)_')
    # Now group by architecture and threshold to get aggregated stats per architecture
    df_thresholds_stats = df_thresholds.groupby(['architecture', 'threshold']).agg(
        micro_f1_mean=('micro_f1', 'mean'),
        macro_f1_mean=('macro_f1', 'mean'),
    ).reset_index()

    plt.figure(figsize=(3.5, 2.5))

    arch_map = {
        "bi": "Bi-Encoder",
        "ce": "Cross-Encoder",
    }

    color_map = {
        "bi": "tab:orange",
        "ce": "tab:blue",
    }

    for arch_key, arch_name in arch_map.items():
        subset = df_thresholds_stats[
            df_thresholds_stats["architecture"] == arch_key
        ].sort_values("threshold")

        plt.plot(
            subset["threshold"],
            subset["micro_f1_mean"],
            marker="o",
            color=color_map[arch_key],
            label=arch_name,
        )

    plt.xlabel("Threshold")
    plt.ylabel("Micro F1")
    plt.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig("thresholds_micro_f1.png", dpi=300)

def architecture_ablation(df):
    df_agg = df[~df['evaluation_dir'].str.contains('translated')]
    df_agg = df_agg.groupby(['experiment']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg = df_agg[df_agg["experiment"].str.contains('pilener')]
    df_agg['architecture'] = df_agg['experiment'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df_agg['experiment'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')

    df_agg = df_agg.pivot(index='architecture', columns='backbone', values='micro_f1')
    df_agg = df_agg.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(df_agg.to_latex())

def dataset_ablation(df):
    df_agg = df[~df['evaluation_dir'].str.contains('translated')]
    df_agg = df_agg.groupby(['experiment']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg = df_agg[df_agg["experiment"].isin(['bi_mmbert_finerweb', 'bi_mmbert_euroglinerx', 'bi_mmbert_pilener', 'ce_xlmr_finerweb', 'ce_xlmr_euroglinerx', 'ce_xlmr_pilener'])]
    df_agg['architecture'] = df_agg['experiment'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df_agg['experiment'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')
    df_agg['dataset'] = df_agg['experiment'].str.extract(r'_(finerweb|euroglinerx|pilener)')

    df_agg = df_agg.pivot(index='architecture', columns='dataset', values='micro_f1')
    df_agg = df_agg.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(df_agg.to_latex()) 

def dataset_ablation_translated(df):
    df_agg = df.copy(deep=True)
    df_agg['is_eval_translated'] = df['evaluation_dir'].str.contains('translated')
    df_agg = df_agg[df_agg["experiment"].isin(['bi_mmbert_finerweb', 'bi_mmbert_finerweb-translated', 'ce_xlmr_finerweb', 'ce_xlmr_finerweb-translated'])]
    df_agg = df_agg.groupby(['experiment', 'is_eval_translated']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg['architecture'] = df_agg['experiment'].str.extract(r'^(ce|bi)_')
    df_agg['is_train_translated'] = df_agg['experiment'].str.contains('translated')

    df_agg = df_agg.pivot(index=['architecture', 'is_train_translated'], columns='is_eval_translated', values='micro_f1')
    df_agg = df_agg.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(df_agg.to_latex()) 

def main():
    rows = []
    for experiment_dir in glob.glob("/vol/tmp/goldejon/ner/paper_results/first_experiment/*"):
        for evalulation_dir in glob.glob(experiment_dir + "/*"):
            evalulation_files = glob.glob(evalulation_dir + "/*.json")
            for evalulation_file in evalulation_files:
                with open(evalulation_file, "r") as f:
                    data = json.load(f)
                rows.append({
                    "experiment": experiment_dir.split("/")[-1],
                    "evaluation_dir": evalulation_dir.split("/")[-1],
                    "language": evalulation_file.split("/")[-1].split("_")[0],
                    "threshold": evalulation_file.split("/")[-1].split("_")[1],
                    "micro_f1": data["test_metrics"]["micro"]["f1"],
                    "macro_f1": data["test_metrics"]["macro"]["f1"],
                })
    df = pd.DataFrame(rows)
    df["threshold"] = (
        df["threshold"]
        .astype(str)
        .str.replace(".json", "", regex=False)
        .astype(float)
    )
    df = df[df['threshold'] == 0.1]

    architecture_ablation(df)
    dataset_ablation(df)

if __name__ == "__main__":
    main()