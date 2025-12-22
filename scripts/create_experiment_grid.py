import json

def main():

    default_biencoder_config = "configs/first_experiment/bi_rembert_finerweb.json"
    default_ceencoder_config = "configs/first_experiment/ce_mdeberta_finerweb.json"

    model_lookup = {
        "rembert": "google/rembert",
        "mdeberta": "microsoft/mdeberta-v3-base",
        "xlmr": "FacebookAI/xlm-roberta-base",
        "mmbert": "jhu-clsp/mmBERT-base",
        "mt5": "google/mt5-base",
    }

    dataset_lookup = {
        'finerweb': "/vol/tmp/goldejon/ner/data/finerweb_splitted/*.jsonl",
        'finerweb-translated': "/vol/tmp/goldejon/ner/data/finerweb_translated_splitted/*.jsonl",
        "euroglinerx": "/vol/tmp/goldejon/ner/data/euroglinerx/train_formatted.jsonl",
        "pilener": "/vol/tmp/goldejon/ner/data/pilener/train_formatted.jsonl",
    }

    for model, model_path in model_lookup.items():
        for dataset, dataset_path in dataset_lookup.items():
            config = json.load(open(default_biencoder_config))
            config["run_name"] = f"bi_{model}_{dataset}"
            config["token_encoder"] = model_path
            config["dataset_name"] = dataset
            config["train_file"] = dataset_path
            if model == "mt5":
                config['learning_rate'] = 1e-3
                config['type_encoder_learning_rate'] = 1e-5
                config['linear_layers_learning_rate'] = 1e-5
                config['lr_scheduler_type'] = "constant"
            if model in ["xlmr", "rembert", "mt5", "mdeberta"]:
                config['max_seq_length'] = 512
            else:
                config['max_seq_length'] = 1024
            config["output_dir"] = f"/vol/tmp2/goldejon/paper_experiments/bi_{model}_{dataset}"
            
            with open(f"configs/first_experiment/bi_{model}_{dataset}.json", "w") as f:
                json.dump(config, f, indent=4)

    for model, model_path in model_lookup.items():
        for dataset, dataset_path in dataset_lookup.items():
            config = json.load(open(default_ceencoder_config))
            config['max_steps'] = 50000
            config["run_name"] = f"ce_{model}_{dataset}"
            config["token_encoder"] = model_path
            config["dataset_name"] = dataset
            config["train_file"] = dataset_path
            if model == "mt5":
                config['learning_rate'] = 1e-3
                config['type_encoder_learning_rate'] = 1e-5
                config['linear_layers_learning_rate'] = 1e-5
                config['lr_scheduler_type'] = "constant"
            if model in ["xlmr", "rembert", "mt5", "mdeberta"]:
                config['max_seq_length'] = 512
            else:
                config['max_seq_length'] = 1024
            config["output_dir"] = f"/vol/tmp2/goldejon/paper_experiments/ce_{model}_{dataset}"
            
            with open(f"configs/first_experiment/ce_{model}_{dataset}.json", "w") as f:
                json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()