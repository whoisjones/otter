import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_span_predictions(span_logits, start_mask, end_mask, max_span_width, id2label, threshold=0.5):
    B, T, S, _ = span_logits.shape

    span_probs = sigmoid(span_logits)
    span_preds = np.triu(span_probs > threshold)
    batch_ids, type_ids, start_indexes, end_indexes = np.nonzero(start_mask[..., None] & end_mask[..., None] & span_preds)

    span_widths = end_indexes - start_indexes + 1
    valid_mask = span_widths <= max_span_width
    batch_ids = batch_ids[valid_mask]
    type_ids = type_ids[valid_mask]
    start_indexes = start_indexes[valid_mask]
    end_indexes = end_indexes[valid_mask]

    confidences = span_probs[batch_ids, type_ids, start_indexes, end_indexes]

    order = confidences.argsort()[::-1]
    batch_ids = batch_ids[order].tolist()
    type_ids = type_ids[order].tolist()
    start_indexes = start_indexes[order].tolist()
    end_indexes = end_indexes[order].tolist()
    confidences = confidences[order].tolist()

    predictions = [[] for _ in range(B)]
    used_by_batch = [set() for _ in range(B)]  # Use sets instead of tensors

    for b, t, s, e, c in zip(batch_ids, type_ids, start_indexes, end_indexes, confidences):
        if any(pos in used_by_batch[b] for pos in range(s, e + 1)):
            continue
        predictions[b].append({"start": s, "end": e, "label": id2label[t], "confidence": c})
        used_by_batch[b].update(range(s, e + 1))
    
    return predictions

def compute_compressed_span_predictions(span_logits, span_mask, span_mapping, id2label, threshold=0.5):
    B, C, S = span_logits.shape

    span_probs = sigmoid(span_logits)
    if threshold == "cls":
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
    
    predictions = [[] for _ in range(B)]
    used_by_batch = [set() for _ in range(B)]

    for b, t, s, c in zip(batch_ids, type_ids, span_ids, confidences):
        start, end = span_mapping[b, s].tolist()
        if any(pos in used_by_batch[b] for pos in range(start, end + 1)):
            continue
        predictions[b].append({"start": start, "end": end, "label": id2label[t], "confidence": c})
        used_by_batch[b].update(range(start, end + 1))

    return predictions

def _norm_pred_item(p):
    if isinstance(p, dict):
        return int(p["start"]), int(p["end"]), str(p["label"])
    s, e, l = p[:3]
    return int(s), int(e), str(l)

def compute_tp_fn_fp(predictions: set, labels: set) -> dict:
    if not predictions and not labels:
        return {"tp": 0, "fn": 0, "fp": 0}
    tp = len(predictions & labels)
    fn = len(labels) - tp
    fp = len(predictions) - tp
    return {"tp": tp, "fn": fn, "fp": fp}

def add_batch_metrics(golds, predictions, metrics_by_type):
    """
    golds: List[List[{'start','end','label'}]]
    predictions: List[List[ dict|tuple ]]
    metrics_by_type: defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    """
    for gold_spans, pred_spans in zip(golds, predictions):
        gold_set = {(int(g["start"]), int(g["end"]), str(g["label"])) for g in gold_spans}
        pred_set = {_norm_pred_item(p) for p in pred_spans}

        types = {t for *_, t in gold_set} | {t for *_, t in pred_set}
        for t in types:
            gold_t = {(s, e, tt) for (s, e, tt) in gold_set if tt == t}
            pred_t = {(s, e, tt) for (s, e, tt) in pred_set if tt == t}
            c = compute_tp_fn_fp(pred_t, gold_t)
            m = metrics_by_type[t]
            m["tp"] += c["tp"]
            m["fp"] += c["fp"]
            m["fn"] += c["fn"]

def finalize_metrics(metrics_by_type, id2label=None):
    """Return per-class + micro/macro dicts. id2label optional."""
    per_class = {}
    TP = FP = FN = 0
    for t, m in metrics_by_type.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2*prec*rec / (prec + rec)) if (prec + rec) else 0.0
        name = id2label.get(t, t) if id2label else t
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "precision": prec, "recall": rec, "f1": f1}
        TP += tp; FP += fp; FN += fn

    micro_p = TP / (TP + FP) if (TP + FP) else 0.0
    micro_r = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f = (2*micro_p*micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Macro over classes that appeared (keys of metrics_by_type)
    if per_class:
        macro_p = sum(v["precision"] for v in per_class.values()) / len(per_class)
        macro_r = sum(v["recall"] for v in per_class.values()) / len(per_class)
        macro_f = sum(v["f1"] for v in per_class.values()) / len(per_class)
    else:
        macro_p = macro_r = macro_f = 0.0

    return {
        "per_class": per_class,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                  "tp": TP, "fp": FP, "fn": FN},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f,
                  "num_classes": len(per_class)}
    }