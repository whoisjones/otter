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
    df_thresholds = df[~df['evaluation_dir'].str.contains('translated')]
    df_thresholds = df_thresholds[df_thresholds['experiment'].str.contains('pilener')]
    df_thresholds = df_thresholds.groupby(['experiment', 'evaluation_dir', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_thresholds = df_thresholds.groupby(['experiment', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_thresholds['architecture'] = df_thresholds['experiment'].str.extract(r'^(ce|bi)_')
    df_thresholds['backbone'] = df_thresholds['experiment'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')
    df_thresholds = df_thresholds.groupby(['architecture', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.3), sharey=True)

    arch_order = [("bi", "Bi-Encoder"), ("ce", "Cross-Encoder")]
    backbone_map = {
        "rembert": "RemBERT",
        "mdeberta": "mDeBERTa",
        "xlmr": "XLM-R",
        "mmbert": "mmBERT",
        "mt5": "mT5",
    }

    for ax, (arch_key, arch_name) in zip(axes, arch_order):
        for backbone_key, bb_name in backbone_map.items():
            subset = (
                df_thresholds[
                    (df_thresholds["architecture"] == arch_key)
                    & (df_thresholds["backbone"] == backbone_key)
                ]
                .sort_values("threshold")
            )
            if subset.empty:
                continue

            ax.plot(subset["threshold"], subset["micro_f1"], marker="o", linewidth=1.2, markersize=3, label=bb_name)

            best_idx = subset["micro_f1"].idxmax()
            ax.scatter(
                [subset.loc[best_idx, "threshold"]],
                [subset.loc[best_idx, "micro_f1"]],
                marker="D",
                s=28,
                edgecolors="black",
                linewidths=0.4,
            )

        ax.set_title(arch_name, fontsize=10)
        ax.set_xlabel("Threshold", fontsize=10)
        if ax == axes[0]:
            ax.legend(frameon=False, fontsize=6, loc="best")

    axes[0].set_ylabel("Macro-F1", fontsize=10)
    fig.tight_layout()
    fig.savefig("thresholds_micro_f1.png", dpi=300)
    plt.close(fig)

def architecture_ablation(df):
    df_agg = df.copy(deep=True)
    df_agg = df_agg[~df_agg['evaluation_dir'].str.contains('translated')]
    df_agg = df_agg[df_agg['experiment'].str.contains('pilener')]
    df_agg = df_agg.groupby(['experiment', 'evaluation_dir', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg = df_agg.groupby(['experiment', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg['architecture'] = df_agg['experiment'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df_agg['experiment'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')

    # select max micro f1 per architecture, backbone
    df_agg = df_agg.groupby(['architecture', 'backbone']).agg({
        'micro_f1': 'max',
    }).reset_index()
    df_agg = df_agg.pivot(index='architecture', columns='backbone', values='micro_f1')
    #add mean column
    df_agg['avg'] = df_agg.mean(axis=1)
    # add mean row for each backbone
    df_agg.loc["avg"] = df_agg.mean(axis=0)
    df_agg = df_agg.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(df_agg.to_latex())

def dataset_ablation(df):
    df_agg = df[~df['evaluation_dir'].str.contains('translated')]
    df_agg = df_agg[
        ((df_agg['experiment'].str.startswith('bi_')) & (df_agg['threshold'] == 0.2)) |
        ((~df_agg['experiment'].str.startswith('bi_')) & (df_agg['threshold'] == 0.15))
    ]
    df_agg = df_agg.groupby(['experiment', 'evaluation_dir']).agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg = df_agg.groupby('experiment').agg({
        'micro_f1': 'mean',
    }).reset_index()
    df_agg = df_agg[df_agg["experiment"].isin(['bi_rembert_finerweb', 'bi_rembert_euroglinerx', 'bi_rembert_pilener', 'ce_xlmr_finerweb', 'ce_xlmr_euroglinerx', 'ce_xlmr_pilener'])]
    df_agg['architecture'] = df_agg['experiment'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df_agg['experiment'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')
    df_agg['dataset'] = df_agg['experiment'].str.extract(r'_(finerweb|euroglinerx|pilener)')

    df_agg = df_agg.pivot(index='architecture', columns='dataset', values='micro_f1')
    df_agg = df_agg.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(df_agg.to_latex()) 

def dataset_ablation_translated(df):
    df_agg = df.copy(deep=True)
    df_agg = df_agg[
        ((df_agg['experiment'].str.startswith('bi_')) & (df_agg['threshold'] == 0.2)) |
        ((~df_agg['experiment'].str.startswith('bi_')) & (df_agg['threshold'] == 0.15))
    ]
    df_agg['is_eval_translated'] = df['evaluation_dir'].str.contains('translated')
    df_agg = df_agg[df_agg["experiment"].isin(['bi_rembert_finerweb', 'bi_rembert_finerweb-translated', 'ce_xlmr_finerweb', 'ce_xlmr_finerweb-translated'])]
    df_agg = df_agg.groupby(['experiment', 'evaluation_dir', "is_eval_translated"]).agg({
        'micro_f1': 'mean',
    }).reset_index()
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

    architecture_ablation(df)
    df = df.groupby(['experiment', 'threshold']).agg({
        'micro_f1': 'mean',
    }).reset_index()

    #sort by micro_f1 mean
    df = df.sort_values(by='micro_f1', ascending=False)
    print(df.to_latex())

if __name__ == "__main__":
    main()