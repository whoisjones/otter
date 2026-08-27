import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_span_predictions(span_logits, span_mask, span_mapping, id2label, threshold=0.5):
    batch_size = span_logits.shape[0]

    span_probs = sigmoid(span_logits)
    if threshold == "cls" or threshold == "label_token":
        span_preds = span_probs > span_probs[:, :, 0:1]
    else:
        span_preds = span_probs > threshold
    batch_ids, type_ids, span_ids = np.nonzero(span_mask & span_preds)
    confidences = span_probs[batch_ids, type_ids, span_ids]

    order = confidences.argsort()[::-1]
    batch_ids = batch_ids[order].tolist()
    type_ids = type_ids[order].tolist()
    span_ids = span_ids[order].tolist()
    confidences = confidences[order].tolist()

    predictions = [[] for _ in range(batch_size)]
    used_by_batch = [set() for _ in range(batch_size)]

    for batch_id, type_id, span_id, confidence in zip(batch_ids, type_ids, span_ids, confidences):
        start, end = span_mapping[batch_id, span_id].tolist()
        if any(pos in used_by_batch[batch_id] for pos in range(start, end + 1)):
            continue
        predictions[batch_id].append(
            {"start": start, "end": end, "label": id2label[type_id], "confidence": confidence}
        )
        used_by_batch[batch_id].update(range(start, end + 1))

    return predictions


def normalize_prediction(p):
    if isinstance(p, dict):
        return int(p["start"]), int(p["end"]), str(p["label"])
    start, end, label = p[:3]
    return int(start), int(end), str(label)


def compute_tp_fn_fp(predictions: set, labels: set) -> dict:
    if not predictions and not labels:
        return {"tp": 0, "fn": 0, "fp": 0}
    tp = len(predictions & labels)
    fn = len(labels) - tp
    fp = len(predictions) - tp
    return {"tp": tp, "fn": fn, "fp": fp}


def add_batch_metrics(golds, predictions, metrics_by_type):
    for gold_spans, pred_spans in zip(golds, predictions):
        gold_set = {(int(g["start"]), int(g["end"]), str(g["label"])) for g in gold_spans}
        pred_set = {normalize_prediction(p) for p in pred_spans}

        types = {label for *_, label in gold_set} | {label for *_, label in pred_set}
        for entity_type in types:
            gold_of_type = {span for span in gold_set if span[2] == entity_type}
            pred_of_type = {span for span in pred_set if span[2] == entity_type}
            counts = compute_tp_fn_fp(pred_of_type, gold_of_type)
            totals = metrics_by_type[entity_type]
            totals["tp"] += counts["tp"]
            totals["fp"] += counts["fp"]
            totals["fn"] += counts["fn"]


def finalize_metrics(metrics_by_type, id2label=None):
    per_class = {}
    total_tp = total_fp = total_fn = 0
    for entity_type, totals in metrics_by_type.items():
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        name = id2label.get(entity_type, entity_type) if id2label else entity_type
        per_class[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    if per_class:
        macro_p = sum(v["precision"] for v in per_class.values()) / len(per_class)
        macro_r = sum(v["recall"] for v in per_class.values()) / len(per_class)
        macro_f = sum(v["f1"] for v in per_class.values()) / len(per_class)
    else:
        macro_p = macro_r = macro_f = 0.0

    return {
        "per_class": per_class,
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f,
            "num_classes": len(per_class),
        },
    }
