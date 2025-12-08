from datasets import load_dataset, Dataset
from gliner import GLiNER

MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": -1,
    "multinerd": -1,
    "multiconer_v1": 1000,
    "multiconer_v2": 1000,
    "dynamicner": -1,
    "uner": -1,
}

def transform_bio_to_span(dataset: Dataset) -> list[dict]:
    """
    Convert BIO format annotations to span-based format.
    Assumes the dataset contains "words" and "ner" fields per sample.

    Returns:
        List of dicts: {"tokenized_text": ..., "spans": ...}
    """
    labels = dataset.features["ner"].feature.names
    id2label = {i: label for i, label in enumerate(labels)}
    span_dataset = []

    for sample in dataset:
        spans = []
        start_idx = None
        ent_type = None
        for idx, tag_id in enumerate(sample["ner"]):
            tag = id2label[tag_id]
            if tag == 'O':
                if start_idx is not None:
                    # Close previous entity
                    spans.append((start_idx, idx - 1, ent_type))
                    start_idx = None
                    ent_type = None
            elif tag.startswith('B-'):
                if start_idx is not None:
                    # Close previous entity before starting a new one
                    spans.append((start_idx, idx - 1, ent_type))
                start_idx = idx
                ent_type = tag[2:]  # Remove 'B-'
            elif tag.startswith('I-'):
                if start_idx is None:
                    # I- without B- before, treat as B-
                    start_idx = idx
                    ent_type = tag[2:]
                # else: continue the current entity
        # Handle last span if exists
        if start_idx is not None:
            spans.append((start_idx, len(sample["ner"]) - 1, ent_type))
        span_dataset.append({"tokenized_text": sample["words"], "ner": spans})
    return span_dataset

def transform_jsonl_to_gliner_dataset(dataset: Dataset) -> list[dict]:
    gliner_format = []
    for sample in dataset:
        gliner_format.append({"tokenized_text": sample["tokens"], "ner": [(annotation["start"], annotation["end"] - 1, annotation["label"]) for annotation in sample["token_spans"]]})
    return gliner_format

def classic_evaluation():
    dataset = load_dataset("json", data_files="/vol/tmp/goldejon/ner/data/thainer/test.jsonl")
    test_dataset_with_span_labels = transform_jsonl_to_gliner_dataset(dataset['train'])
    labels = list(set([label for sample in test_dataset_with_span_labels for _, _, label in sample["ner"]]))
    model = GLiNER.from_pretrained("knowledgator/gliner-x-base")
    model.to("cuda")

    results, f1 = model.evaluate(test_dataset_with_span_labels, flat_ner=True, batch_size=12, entity_types=list(labels))
    print(results)

def string_evaluation():
    import glob
    from datasets import DatasetDict
    import os
    import json
    for model_name_or_path in ['urchade/gliner_multi-v2.1', 'knowledgator/gliner-x-base', "knowledgator/gliner-x-large"]:
        model = GLiNER.from_pretrained(model_name_or_path)
        model.to("cuda")
        for path in ["/vol/tmp/goldejon/ner/eval_data"]:
            for dataset_path in glob.glob(path + "/*"):
                for language_dataset_path in glob.glob(dataset_path + "/*"):
                    if os.path.exists(f"ablations/{model_name_or_path.split('/')[-1]}/{dataset_path.split('/')[-1]}/{language_dataset_path.split('/')[-1]}.json"):
                        continue

                    dataset = DatasetDict.load_from_disk(language_dataset_path)
                    if "test"  in dataset:
                        test_split = dataset["test"]
                    else:
                        test_split = dataset["dev"]

                    labels = list(set([span["label"] for sample in test_split for span in sample['char_spans']]))

                    # Overall metrics (micro-averaged)
                    tp, fp, fn = 0, 0, 0
                    # Per-class metrics
                    per_class_metrics = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
                    
                    for sample in test_split:
                        preds = model.predict_entities(sample['text'], flat_ner=True, labels=list(labels))
                        golds = sample['char_spans']
                        preds = [(pred['start'], pred['end'], pred['label']) for pred in preds]
                        golds = [(gold['start'], gold['end'], gold['label']) for gold in golds]
                        
                        # Overall metrics
                        tp += len(set(preds) & set(golds))
                        fp += len(set(preds) - set(golds))
                        fn += len(set(golds) - set(preds))
                        
                        # Per-class metrics
                        for label in labels:
                            preds_label = {(s, e, l) for s, e, l in preds if l == label}
                            golds_label = {(s, e, l) for s, e, l in golds if l == label}
                            per_class_metrics[label]["tp"] += len(preds_label & golds_label)
                            per_class_metrics[label]["fp"] += len(preds_label - golds_label)
                            per_class_metrics[label]["fn"] += len(golds_label - preds_label)

                    # Overall (micro-averaged) metrics
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    
                    # Per-class metrics
                    per_class_results = {}
                    for label, metrics in per_class_metrics.items():
                        tp_cls = metrics["tp"]
                        fp_cls = metrics["fp"]
                        fn_cls = metrics["fn"]
                        p_cls = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
                        r_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0
                        f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0
                        per_class_results[label] = {
                            "precision": p_cls,
                            "recall": r_cls,
                            "f1": f1_cls,
                            "tp": tp_cls,
                            "fp": fp_cls,
                            "fn": fn_cls
                        }
                    
                    os.makedirs(f"ablations/{model_name_or_path.split('/')[-1]}/{dataset_path.split('/')[-1]}", exist_ok=True)
                    with open(f"ablations/{model_name_or_path.split('/')[-1]}/{dataset_path.split('/')[-1]}/{language_dataset_path.split('/')[-1]}.json", "w") as f:
                        json.dump({
                            "micro_average": {
                                "precision": p,
                                "recall": r,
                                "f1": f1,
                                "tp": tp,
                                "fp": fp,
                                "fn": fn
                            },
                            "per_class": per_class_results
                        }, f, indent=2)
    
if __name__ == "__main__":
    string_evaluation()