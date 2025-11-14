#!/usr/bin/env python3
"""
Main training script for Dual Encoder NER model.
Run from project root: python train.py --config configs/default.json
"""

import os
import sys
import json
import warnings
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import SpanModel 
from src.config import SpanModelConfig
from src.collator import AllLabelsDataCollator
from src.trainer import evaluate
from src.logger import setup_logger
from src.args import ModelArguments, DataTrainingArguments

transformers.logging.set_verbosity_error()

pretrained_model_name_or_path = "/vol/tmp/goldejon/ner/finerweb-multi/checkpoint-5000"
test_file = "/vol/tmp/goldejon/ner/data/thainer_no_tokens/test.jsonl"

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    logger = setup_logger(training_args.output_dir)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    torch.manual_seed(training_args.seed)
    
    # Initialize accelerator for mixed precision and device management
    accelerator = Accelerator(
        mixed_precision="fp16" if getattr(training_args, 'fp16', False) else "no",
        gradient_accumulation_steps=getattr(training_args, 'gradient_accumulation_steps', 1)
    )
    
    data_files = {}
    data_files["test"] = test_file
    dataset = load_dataset('json', data_files=data_files)

    config = SpanModelConfig.from_pretrained(pretrained_model_name_or_path)
    model = SpanModel(config=config)
    model = model.from_pretrained(pretrained_model_name_or_path)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)

    if "test" not in dataset:
        raise ValueError("--do_predict requires a test file.")
    test_labels = list(set([span["label"] for sample in dataset["test"] for span in sample["token_spans" if data_args.annotation_format == "tokens" else "char_spans"]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}
    type_encodings = type_encoder_tokenizer(
        list(label2id.keys()),
        truncation=True,
        max_length=64,
        padding="longest" if len(test_labels) <= 1000 else "max_length",
        return_tensors="pt"
    )
    test_collator = AllLabelsDataCollator(
        token_encoder_tokenizer, 
        type_encodings=type_encodings,
        label2id=label2id,
        max_seq_length=data_args.max_seq_length, 
        format=data_args.annotation_format,
        loss_masking=data_args.loss_masking
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=training_args.per_device_eval_batch_size,
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
        
    test_results_path = Path(training_args.output_dir) / "test_results_swa.json"
    with open(test_results_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
            "config": config.to_dict(),
        }, f, indent=2)
    logger.info(f"\nTest results saved to {test_results_path}")


if __name__ == "__main__":
    main()

