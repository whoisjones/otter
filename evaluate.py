import os
import argparse
from datetime import datetime
import json
import warnings

import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, get_dataset_config_names
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import (
    OtterBiEncoderModel,
    OtterCrossEncoderModel,
    OtterContrastiveBiEncoderModel,
    OtterContrastiveCrossEncoderModel,
)
from src.config import SpanModelConfig
from src.collator import (
    EvalCollatorBiEncoder,
    EvalCollatorCrossEncoder,
    EvalCollatorContrastiveBiEncoder,
    EvalCollatorContrastiveCrossEncoder,
)
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()

ARCHITECTURE_REGISTRY = {
    "bi_encoder": (OtterBiEncoderModel, EvalCollatorBiEncoder),
    "cross_encoder": (OtterCrossEncoderModel, EvalCollatorCrossEncoder),
    "contrastive_bi_encoder": (OtterContrastiveBiEncoderModel, EvalCollatorContrastiveBiEncoder),
    "contrastive_cross_encoder": (OtterContrastiveCrossEncoderModel, EvalCollatorContrastiveCrossEncoder),
}


def _is_bi_encoder(architecture: str) -> bool:
    return "bi_encoder" in architecture


def _is_contrastive(architecture: str) -> bool:
    return "contrastive" in architecture


def run_eval(
    pretrained_model_name_or_path,
    dataset,
    result_save_path,
    prediction_threshold=None,
    evaluation_format="text",
    identifier=None,
):
    logger = setup_logger("evaluate")
    torch.manual_seed(42)

    accelerator = Accelerator(mixed_precision="bf16")

    config = SpanModelConfig.from_pretrained(pretrained_model_name_or_path)

    # Determine architecture: prefer custom field, fall back to HF architectures list
    _HF_CLASS_TO_ARCH = {
        "OtterBiEncoderModel": "bi_encoder",
        "OtterCrossEncoderModel": "cross_encoder",
        "OtterContrastiveBiEncoderModel": "contrastive_bi_encoder",
        "OtterContrastiveCrossEncoderModel": "contrastive_cross_encoder",
    }
    architecture = getattr(config, "architecture", None)
    if architecture is None:
        hf_archs = getattr(config, "architectures", None) or []
        for cls_name in hf_archs:
            if cls_name in _HF_CLASS_TO_ARCH:
                architecture = _HF_CLASS_TO_ARCH[cls_name]
                break
    if architecture is None:
        raise ValueError(
            "Cannot determine architecture from checkpoint. "
            "Re-train with the unified train.py which stores 'architecture' in the config."
        )

    if architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    model_cls, eval_collator_cls = ARCHITECTURE_REGISTRY[architecture]
    bi_encoder = _is_bi_encoder(architecture)
    contrastive = _is_contrastive(architecture)

    model = model_cls.from_pretrained(pretrained_model_name_or_path)

    # Load tokenizer(s)
    if bi_encoder:
        token_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
        type_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
    else:
        # Try checkpoint root first (best_checkpoint_hf), then tokenizer/ subdir, then base model
        tokenizer_path = pretrained_model_name_or_path
        subdir_path = os.path.join(pretrained_model_name_or_path, "tokenizer")
        if os.path.exists(subdir_path):
            tokenizer_path = subdir_path
        try:
            token_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            token_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
            token_tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
            if contrastive:
                token_tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)

    # Resolve prediction threshold
    collator_threshold = None
    if prediction_threshold is not None:
        try:
            model.config.prediction_threshold = float(prediction_threshold)
        except (ValueError, TypeError):
            if prediction_threshold in ("cls", "label_token"):
                model.config.prediction_threshold = prediction_threshold
                collator_threshold = prediction_threshold
            else:
                raise ValueError(
                    f"Invalid threshold '{prediction_threshold}'. "
                    "Must be a float, 'cls', or 'label_token'."
                )
    else:
        default = getattr(model.config, "prediction_threshold", 0.5)
        if isinstance(default, str):
            collator_threshold = default

    if contrastive and not bi_encoder and collator_threshold is None:
        collator_threshold = "label_token"

    # Normalise dataset column names if needed
    if "spans_tokens" in dataset.column_names:
        dataset = dataset.rename_column("spans_tokens", "token_spans")
    if "spans_char" in dataset.column_names:
        dataset = dataset.rename_column("spans_char", "char_spans")

    def _rename_annotation_key(sample):
        return {
            "char_spans": [
                {"start": s["start"], "end": s["end"], "label": s.get("label", s.get("tag"))}
                for s in sample["char_spans"]
            ],
            "token_spans": [
                {"start": s["start"], "end": s["end"], "label": s.get("label", s.get("tag"))}
                for s in sample["token_spans"]
            ],
        }

    if dataset.column_names and "char_spans" in dataset.column_names:
        first = dataset[0]["char_spans"]
        if first and "tag" in first[0]:
            dataset = dataset.map(_rename_annotation_key)

    span_key = "token_spans" if evaluation_format == "tokens" else "char_spans"
    test_labels = list(set(span["label"] for sample in dataset for span in sample[span_key]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}

    max_seq_length = 1024 if config.token_encoder == "jhu-clsp/mmBERT-base" else 512
    loss_masking = "none" if evaluation_format == "text" else "subwords"

    if bi_encoder:
        type_encodings = type_tokenizer(
            list(label2id.keys()),
            truncation=True,
            max_length=64,
            padding="longest" if len(test_labels) <= 1000 else "max_length",
            return_tensors="pt",
        )
        test_collator = eval_collator_cls(
            token_tokenizer,
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=max_seq_length,
            format=evaluation_format,
            loss_masking=loss_masking,
        )
    else:
        kwargs = dict(
            label2id=label2id,
            max_seq_length=max_seq_length,
            format=evaluation_format,
            loss_masking=loss_masking,
        )
        if contrastive:
            kwargs["prediction_threshold"] = collator_threshold or "label_token"
        test_collator = eval_collator_cls(token_tokenizer, **kwargs)

    test_dataloader = DataLoader(
        dataset,
        batch_size=1 if not bi_encoder else 12,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    logger.info("\n" + "=" * 60)
    logger.info("Test Set Evaluation")
    logger.info("=" * 60)
    test_metrics = evaluate(model, test_dataloader, accelerator, track_benchmark=True)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Precision: {test_metrics['micro']['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['micro']['recall']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['micro']['f1']:.4f}")

    if "benchmark" in test_metrics:
        b = test_metrics["benchmark"]
        logger.info("-" * 60)
        if "cuda_memory_allocated_mb" in b:
            logger.info(f"VRAM: {b['cuda_memory_allocated_mb']:.2f} MB")
        if "latency_ms_per_example" in b:
            logger.info(f"Latency: {b['latency_ms_per_example']:.2f} ms/example")
        if "flops_per_forward" in b:
            f = b["flops_per_forward"]
            logger.info(f"FLOPs: {f / 1e9:.2f} G" if f is not None else "FLOPs: N/A")
    logger.info("=" * 60)

    if identifier is not None:
        test_metrics["eval_from"] = identifier

    with open(result_save_path, "w") as fh:
        json.dump({"test_metrics": test_metrics}, fh, indent=2)
    logger.info(f"\nTest results saved to {result_save_path}")


def main(args):
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required")

    if args.evaluation_dataset.endswith(".jsonl"):
        test_splits = {
            args.evaluation_dataset: load_dataset("json", data_files=args.evaluation_dataset, split="train")
        }
    elif os.path.exists(args.evaluation_dataset) and os.path.isdir(args.evaluation_dataset):
        ds = DatasetDict.load_from_disk(args.evaluation_dataset)
        split = "test" if "test" in ds else "dev"
        test_splits = {args.evaluation_dataset: ds[split]}
    else:
        config_names = get_dataset_config_names(args.evaluation_dataset)
        test_splits = {}
        for name in config_names:
            ds = load_dataset(args.evaluation_dataset, name)
            split = "test" if "test" in ds else "dev"
            test_splits[name] = ds[split]

    for identifier, test_split in test_splits.items():
        if len(test_split) > args.max_eval_samples:
            test_split = test_split.shuffle(seed=42).select(range(args.max_eval_samples))
        os.makedirs("evals", exist_ok=True)
        result_save_path = os.path.join("evals", datetime.now().strftime("eval_%Y%m%d_%H%M%S.json"))
        run_eval(
            args.pretrained_model_name_or_path,
            test_split,
            result_save_path,
            args.threshold,
            args.evaluation_format,
            identifier,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--evaluation_format", type=str, choices=["text", "tokens"], default="text")
    args = parser.parse_args()
    main(args)
