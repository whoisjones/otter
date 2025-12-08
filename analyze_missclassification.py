#!/usr/bin/env python3
"""
Main training script for Dual Encoder NER model.
Run from project root: python train.py --config configs/default.json
"""

import os
import glob
import json
import warnings
from collections import defaultdict
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer
from datasets import DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import CompressedSpanModel, ContrastiveSpanModel 
from src.config import SpanModelConfig
from src.collator import AllLabelsCompressedSpanCollator, AllLabelsContrastiveDataCollator
from src.logger import setup_logger

transformers.logging.set_verbosity_error()

pretrained_model_name_or_path = "/vol/tmp/goldejon/ner/finerweb-rembert-contrastive/checkpoint-55000"
eval_dataset = "/vol/tmp/goldejon/ner/eval_data/masakhaner/*"

MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": -1,
    "multinerd": -1,
    "multiconer_v1": 10000,
    "multiconer_v2": 10000,
    "dynamicner": -1,
    "uner": -1,
}

def main():
    # run_analysis()
    validation_split = []
    for dataset in glob.glob("/vol/tmp/goldejon/ner/eval_data/masakhaner/*"):
        dataset = DatasetDict.load_from_disk(dataset)
        if "test" in dataset:
            test_split = dataset["test"]
        else:
            test_split = dataset["dev"]
        test_split = test_split.shuffle(seed=42).select(range(50))
        validation_split.append(test_split)
    
    for dataset in glob.glob("/vol/tmp/goldejon/ner/eval_data/multinerd/*"):
        dataset = DatasetDict.load_from_disk(dataset)
        if "test" in dataset:
            test_split = dataset["test"]
        else:
            test_split = dataset["dev"]
        test_split = test_split.shuffle(seed=42).select(range(50))
        validation_split.append(test_split)

    from datasets import concatenate_datasets
    validation_split = concatenate_datasets(validation_split)
    validation_split = validation_split.remove_columns(["tokens", "token_spans"])
    validation_split = validation_split.add_column("id", list(range(len(validation_split))))
    validation_split.to_json("/vol/tmp/goldejon/ner/data/validation_split.jsonl", orient="records")

def compute_compressed_span_predictions(span_logits, span_mask, span_mapping, id2label, threshold=0.5):
    B, C, S = span_logits.shape

    span_probs = torch.sigmoid(span_logits)
    if threshold == "cls":
        span_preds = span_probs > span_probs[:, :, 0:1] - 1e-4
    else:
        span_preds = span_probs > threshold
    batch_ids, type_ids, span_ids = (span_mask & span_preds).nonzero(as_tuple=True)
    confidences = span_probs[batch_ids, type_ids, span_ids]

    order = torch.argsort(confidences, descending=True)
    batch_ids = batch_ids[order]
    type_ids = type_ids[order]
    span_ids = span_ids[order]
    confidences = confidences[order]

    # Convert to CPU and lists early to avoid GPU memory accumulation
    batch_ids_list = batch_ids.cpu().tolist()
    type_ids_list = type_ids.cpu().tolist()
    span_ids_list = span_ids.cpu().tolist()
    confidences_list = confidences.cpu().tolist()
    span_mapping_cpu = span_mapping.cpu()
    span_lookup = {(start, end): span_idx for span_idx, (start, end) in enumerate(span_mapping_cpu[0].tolist())}
    
    predictions = [[] for _ in range(B)]
    used_by_batch = [set() for _ in range(B)]

    for b, t, s, c in zip(batch_ids_list, type_ids_list, span_ids_list, confidences_list):
        start, end = span_mapping_cpu[b, s].tolist()
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

def evaluate(model, dataloader, accelerator):
    """Evaluate the model on a dataset."""
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    total_loss = 0.0
    num_batches = 0
    
    metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), desc="Evaluating", disable=not accelerator.is_local_main_process)
        for batch in dataloader:
            output = model(
                token_encoder_inputs=batch["token_encoder_inputs"],
                type_encoder_inputs=batch["type_encoder_inputs"],
                labels=batch["labels"]
            )

            if "loss" in output:
                loss = output.loss
                total_loss += loss.detach().item()
                num_batches += 1

            golds = batch['labels']['ner']
            predictions = compute_compressed_span_predictions(
                span_logits=output.span_logits.detach(),
                span_mask=batch["labels"]["valid_span_mask"],
                span_mapping=batch["labels"]["span_subword_indices"],
                id2label=batch["id2label"],
                threshold=unwrapped_model.config.prediction_threshold
            )
            add_batch_metrics(golds, predictions, metrics_by_type)
            
            pbar.update(1)
        pbar.close()

    if total_loss > 0:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = finalize_metrics(metrics_by_type)
        metrics["loss"] = avg_loss
    else:
        metrics = finalize_metrics(metrics_by_type)
        metrics["loss"] = 0.0

    torch.cuda.empty_cache()
    
    return metrics

def run_analysis():
    model_name = pretrained_model_name_or_path.split("/")[-2]
    for language_dataset in glob.glob(eval_dataset):
        dataset = DatasetDict.load_from_disk(language_dataset)

        eval_split = "test" if "test" in dataset else "dev"
        test_split = dataset[eval_split]

        max_samples = MAX_EVAL_SAMPLES_PER_DATASET[language_dataset.split("/")[-2]]
        if max_samples == -1 or max_samples > len(test_split):
            test_split = test_split.shuffle(seed=42)
        else:
            test_split = test_split.shuffle(seed=42).select(range(max_samples))
        os.makedirs(f"analysis/{model_name}/{language_dataset.split("/")[-2]}", exist_ok=True)
        result_save_path = f"analysis/{model_name}/{language_dataset.split("/")[-2]}/{language_dataset.split("/")[-1]}.json"
        run_eval(test_split, pretrained_model_name_or_path, result_save_path)

def run_eval(dataset, pretrained_model_name_or_path, result_save_path):
    logger = setup_logger('eval.log')
    logger.warning(
        f"Process rank: {0}, device: cuda, n_gpu: 1, "
        + f"distributed training: False, 16-bits training: True"
    )
    
    torch.manual_seed(42)
    
    accelerator = Accelerator(
        mixed_precision="bf16"
    )
    
    config = SpanModelConfig.from_pretrained(pretrained_model_name_or_path)
    if config.loss_fn == "contrastive":
        model = ContrastiveSpanModel(config=config).to("cuda")
    else:
        model = CompressedSpanModel(config=config).to("cuda")
    model = model.from_pretrained(pretrained_model_name_or_path)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)

    test_labels = list(set([span["label"] for sample in dataset for span in sample["token_spans"]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}
    type_encodings = type_encoder_tokenizer(
        list(label2id.keys()),
        truncation=True,
        max_length=64,
        padding="longest" if len(test_labels) <= 1000 else "max_length",
        return_tensors="pt"
    )
    if config.loss_fn == "contrastive":
        test_collator = AllLabelsContrastiveDataCollator(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format="tokens",
            loss_masking="subwords"
        )
    else:
        test_collator = AllLabelsCompressedSpanCollator(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format="tokens",
            loss_masking="subwords"
        )
    test_dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 60)
    test_metrics = evaluate(model, test_dataloader, accelerator)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Precision: {test_metrics['micro']['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['micro']['recall']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['micro']['f1']:.4f}")
    logger.info("=" * 60)
        
    with open(result_save_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
        }, f, indent=2)
    logger.info(f"\nTest results saved to {result_save_path}")


if __name__ == "__main__":
    main()

