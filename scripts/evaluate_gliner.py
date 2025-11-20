from datasets import load_dataset, Dataset
from gliner import GLiNER

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
    dataset = load_dataset("json", data_files="/vol/tmp/goldejon/ner/data/thainer_no_tokens/test.jsonl")
    labels = list(set([span["label"] for sample in dataset['train'] for span in sample['char_spans']]))
    model = GLiNER.from_pretrained("knowledgator/gliner-x-base")
    model.to("cuda")

    tp, fp, fn = 0, 0, 0
    for sample in dataset['train']:
        preds = model.predict_entities(sample['text'], flat_ner=True, labels=list(labels))
        golds = sample['char_spans']
        preds = [(pred['start'], pred['end'], pred['label']) for pred in preds]
        golds = [(gold['start'], gold['end'], gold['label']) for gold in golds]
        tp += len(set(preds) & set(golds))
        fp += len(set(preds) - set(golds))
        fn += len(set(golds) - set(preds))
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print(f"p: {p:.4f}, r: {r:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    classic_evaluation()