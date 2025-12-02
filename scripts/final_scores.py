import glob
import json
import pandas as pd

def main():
    rows = []
    for model in glob.glob('results/*'):
        for eval_dataset in glob.glob(model + '/*'):
            for language in glob.glob(eval_dataset + '/*'):
                with open(language, 'r') as f:
                    data = json.load(f)
                    rows.append({
                        'model': model.split('/')[-1],
                        'eval_dataset': eval_dataset.split('/')[-1],
                        'language': language.split('/')[-1].split('.')[0],
                        'micro_precision': data['test_metrics']['micro']['precision'],
                        'micro_recall': data['test_metrics']['micro']['recall'],
                        'micro_f1': data['test_metrics']['micro']['f1'],
                        'macro_precision': data['test_metrics']['macro']['precision'],
                        'macro_recall': data['test_metrics']['macro']['recall'],
                        'macro_f1': data['test_metrics']['macro']['f1'],
                    })

    df = pd.DataFrame(rows)
    # agg over languages that we have macro averaged scroes per model and eval_dataset
    df = df.groupby(['model', 'eval_dataset']).agg({
        'micro_precision': 'mean',
        'micro_recall': 'mean',
        'micro_f1': 'mean',
        'macro_precision': 'mean',
        'macro_recall': 'mean',
        'macro_f1': 'mean',
    }).reset_index()
    
    for metric in ['micro_f1']:
        pivot_df = df.pivot(index='model', columns='eval_dataset', values=metric)
        pivot_df['average'] = pivot_df.mean(axis=1)
        print(pivot_df.to_markdown())

if __name__ == "__main__":
    main()