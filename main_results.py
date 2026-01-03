import glob
import json
import pandas as pd
import os

def baselines():
    results = []
    for result_dir in ["/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner_multi-v2.1/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner-x-base/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner-x-large/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gemma3-27b",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/qwen3-32b",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gpt5",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/wikineural"]:
        model_name = result_dir.split("/")[-2] if "gliner" in result_dir else result_dir.split("/")[-1]
        for dataset_dir in os.listdir(result_dir):
            for language_result_file in os.listdir(f"{result_dir}/{dataset_dir}"):
                language_code = language_result_file.split(".")[0]
                with open(f"{result_dir}/{dataset_dir}/{language_result_file}", "r") as f:
                    result = json.load(f)

                if 'f1' in result:
                    f1 = result['f1']
                else:
                    f1 = result['overall']['f_score']

                results.append({
                    'model': model_name if "evaluation_translated" not in result_dir else model_name + " (translated)",
                    'dataset': dataset_dir,
                    'language': language_code,
                    'f1': f1
                }) 
    
    df = pd.DataFrame(results)
    return df

def main():
    baseline_df = baselines()

    rows = []
    for result_dir in glob.glob("/vol/tmp/goldejon/ner/paper_results/first_experiment/*"):
        for evaluation_dir in glob.glob(result_dir + "/*"):
            for evaluation_file in glob.glob(evaluation_dir + "/*.json"):
                with open(evaluation_file, "r") as f:
                    data = json.load(f)
                rows.append({
                    "model": result_dir.split("/")[-1],
                    "dataset": evaluation_dir.split("/")[-1],
                    "language": evaluation_file.split("/")[-1].split("_")[0],
                    "threshold": evaluation_file.split("/")[-1].split("_")[1].split(".j")[0],
                    "f1": data["test_metrics"]["micro"]["f1"],
                })

    df = pd.DataFrame(rows)
    df["threshold"] = (
        df["threshold"]
        .astype(float)
    )
    df = df[~df['dataset'].str.contains('translated')]
    df = df[~df['model'].str.contains('translated')]
    df_agg = df.copy(deep=True)
    df_agg['architecture'] = df['model'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df['model'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')
    df_agg['train_dataset'] = df['model'].str.extract(r'_(finerweb-translated|euroglinerx|pilener|finerweb)')
    df_lang_f1 = df_agg.groupby(["architecture", "threshold"]).agg({"f1": "mean"}).reset_index()
    df_dataset_f1 = df_agg.groupby(["architecture", 'backbone', 'train_dataset', "threshold", "dataset"]).agg({"f1": "mean"}).reset_index()
    df_dataset_f1 = df_dataset_f1.pivot(index=["architecture", 'backbone', 'train_dataset', "threshold"], columns="dataset", values="f1")
    df_dataset_f1["average"] = df_dataset_f1.mean(axis=1)
    # 3 decimal places
    df_dataset_f1 = df_dataset_f1.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    arch_order = ["bi", "ce"]
    ds_order   = ["pilener", "euroglinerx", "finerweb"]

    archtecture_name = {"bi": "Bi-Encoder", "ce": "Cross-Encoder"}
    eval_benchmarks = {
        "panx": "PAN-X",
        "masakhaner": "MasakhaNER",
        "multinerd": "MultiNERD",
        "multiconer_v1": "MultiCoNER v1",
        "multiconer_v2": "MultiCoNER v2",
        "dynamicner": "DynamicNER",
        "uner": "UNER",
    }
    train_datasets = {
        "pilener": "PileNER",
        "euroglinerx": "EuroGLINER-x",
        "finerweb": "FiNERWeb",
    }

    def rot(s: str) -> str:
        # width controls the parbox height after rotation; adjust if needed
        return rf"\rotatebox{{90}}{{{{\centering {s}}}}}"
        
    for m in df_dataset_f1.index.get_level_values("backbone").unique():
        df_filtered = df_dataset_f1.xs(m, level="backbone")

        # figure out the first three remaining index levels after xs()
        idx_names = list(df_filtered.index.names)
        arch_lvl, ds_lvl, thr_lvl = idx_names[0], idx_names[1], idx_names[2]

        # sort by (architecture, train_dataset, threshold asc), keeping index
        df_sorted = (
            df_filtered
            .reset_index()
            .assign(
                **{
                    arch_lvl: lambda d: pd.Categorical(d[arch_lvl], categories=arch_order, ordered=True),
                    ds_lvl:   lambda d: pd.Categorical(d[ds_lvl],   categories=ds_order,   ordered=True),
                    thr_lvl:  lambda d: pd.to_numeric(d[thr_lvl].astype(str).str.replace(".json", "", regex=False)),
                }
            )
            .sort_values([arch_lvl, ds_lvl, thr_lvl], ascending=[True, True, True])
            .set_index([arch_lvl, ds_lvl, thr_lvl] + idx_names[3:])  # keep any extra remaining levels
        )

        # (optional) if there are extra index levels beyond thr, you probably want to drop them here:
        # df_sorted = df_sorted.droplevel(idx_names[3:])

        df_out = df_sorted.rename(columns=eval_benchmarks)

        # Avg column last (numeric)
        df_out["Avg."] = df_out.mean(axis=1)

        # rename index level *values* (still MultiIndex)
        df_out = (
            df_out
            .rename(index=archtecture_name, level=0)   # architecture values
            .rename(index=train_datasets,    level=1)  # train_dataset values
        )

        # rotate the first two index levels (MultiIndex) so multirow still works
        df_out = df_out.rename(index=lambda v: rot(str(v)), level=0)
        df_out = df_out.rename(index=lambda v: rot(str(v)), level=1)
        df_out = df_out.rename(index=lambda v: f"{float(v):.3f}", level=2)

        # name the index columns (these become the first 3 header cells)
        df_out.index = df_out.index.set_names(["Arc.", "Dataset", r"$\tau$"])

        # enforce column order
        bench_cols = ["DynamicNER", "MasakhaNER", "MultiCoNER v1", "MultiCoNER v2", "MultiNERD", "PAN-X", "UNER"]
        df_out = df_out[bench_cols + ["Avg."]]

        # LaTeX column format includes the 3 index columns:
        col_format = "lll|" + "c"*len(bench_cols) + "|r"

        print(
            df_out.to_latex(
                escape=False,          # keep rotatebox/parbox
                multirow=True,         # collapses repeated index values using \multirow
                multicolumn=False,
                column_format=col_format,
                float_format="%.3f",   # replaces applymap for numeric formatting
                caption=f"Results for $m={m}$",  # dynamically include value of m
            )
        )


if __name__ == "__main__":
    main()