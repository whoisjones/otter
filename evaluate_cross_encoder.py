import os
import argparse
from datetime import datetime
import json
import warnings

import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import OtterCrossEncoderModel, OtterContrastiveCrossEncoderModel 
from src.config import SpanModelConfig
from src.collator import EvalCollatorCrossEncoder, EvalCollatorContrastiveCrossEncoder
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()

def run_eval(pretrained_model_name_or_path, dataset, result_save_path, prediction_threshold=None, evaluation_format="text"):
    logger = setup_logger('eval_ce')
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
        model = OtterContrastiveCrossEncoderModel(config=config).to("cuda")
    else:
        model = OtterCrossEncoderModel(config=config).to("cuda")
    model = model.from_pretrained(pretrained_model_name_or_path)

    try:
        tokenizer_path = os.path.join(pretrained_model_name_or_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            token_encoder_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise FileNotFoundError("Tokenizer not found in checkpoint")
    except Exception:
        token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
        if config.loss_fn == "contrastive":
            token_encoder_tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
            token_encoder_tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)
        else:
            token_encoder_tokenizer.add_tokens(["[LABEL]"], special_tokens=True)

    if "spans_tokens" in dataset.column_names:
        dataset = dataset.rename_column("spans_tokens", "token_spans")
    if "spans_char" in dataset.column_names:
        dataset = dataset.rename_column("spans_char", "char_spans")

    def rename_annotation_key(sample):
        token_spans = [{'start': span['start'], 'end': span['end'], 'label': span['tag']} for span in sample['token_spans']]
        char_spans = [{'start': span['start'], 'end': span['end'], 'label': span['tag']} for span in sample['char_spans']]
        return {
            "char_spans": char_spans,
            "token_spans": token_spans
        }
    dataset = dataset.map(rename_annotation_key)
    span_key = "token_spans" if evaluation_format == "tokens" else "char_spans"
    test_labels = list(set([span["label"] for sample in dataset for span in sample[span_key]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}

    if prediction_threshold is not None:
        try:
            threshold_value = float(prediction_threshold)
            model.config.prediction_threshold = threshold_value
            collator_threshold = None
        except (ValueError, TypeError):
            if prediction_threshold in ["cls", "label_token"]:
                model.config.prediction_threshold = prediction_threshold
                collator_threshold = prediction_threshold
            else:
                raise ValueError(f"Invalid threshold: {prediction_threshold}. Must be float, 'cls', or 'label_token'")
    else:
        default_threshold = getattr(model.config, 'prediction_threshold', 0.5)
        if isinstance(default_threshold, str):
            collator_threshold = default_threshold
        else:
            collator_threshold = None

    max_seq_length = 512 if config.token_encoder != "jhu-clsp/mmBERT-base" else 1024
    loss_masking = "none" if evaluation_format == "text" else "subwords"

    if config.loss_fn == "contrastive":
        if collator_threshold is None:
            collator_threshold = "label_token"
        test_collator = EvalCollatorContrastiveCrossEncoder(
            token_encoder_tokenizer, 
            label2id=label2id,
            max_seq_length=max_seq_length, 
            format=evaluation_format,
            loss_masking=loss_masking,
            prediction_threshold=collator_threshold
        )
    else:
        test_collator = EvalCollatorCrossEncoder(
            token_encoder_tokenizer, 
            label2id=label2id,
            max_seq_length=max_seq_length, 
            format=evaluation_format,
            loss_masking=loss_masking
        )
    test_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Set Evaluation")
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
            logger.info(f"FLOPs: {f/1e9:.2f} G" if f is not None else "FLOPs: N/A")
    logger.info("=" * 60)
        
    with open(result_save_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
        }, f, indent=2)
    logger.info(f"\nTest results saved to {result_save_path}")

def run_single_eval(pretrained_model_name_or_path, evaluation_file, threshold, evaluation_format):
    if evaluation_file.endswith(".jsonl"):
        test_split = load_dataset('json', data_files=evaluation_file, split="train")
    else:
        dataset = DatasetDict.load_from_disk(evaluation_file)
        eval_split = "test" if "test" in dataset else "dev"
        test_split = dataset[eval_split].shuffle(seed=42).select(range(1000))
    os.makedirs("evals", exist_ok=True)
    result_save_path = os.path.join("evals", datetime.now().strftime("eval_%Y%m%d_%H%M%S.json"))
    run_eval(pretrained_model_name_or_path, test_split, result_save_path, threshold, evaluation_format)

def main(args):
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required when evaluating a single model")
    run_single_eval(args.pretrained_model_name_or_path, args.evaluation_file, args.threshold, args.evaluation_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--evaluation_format", type=str, choices=["text", "tokens"], default="text")
    args = parser.parse_args()
    main(args)

