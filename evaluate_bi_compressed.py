#!/usr/bin/env python3
"""
Main training script for Dual Encoder NER model.
Run from project root: python train.py --config configs/default.json
"""

import os
import sys
import glob
import json
import warnings
from pathlib import Path

import torch
import transformers
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup, HfArgumentParser, TrainingArguments)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import CompressedBiEncoderModel, ContrastiveBiEncoderModel 
from src.config import SpanModelConfig
from src.collator import EvalCollatorCompressedBiEncoder, EvalCollatorContrastiveBiEncoder
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()

pretrained_model_name_or_paths = [
    "/vol/tmp/goldejon/ner/finerweb-rembert-bce-linear/best_checkpoint",
]
test_file = "/vol/tmp/goldejon/ner/data/thainer_no_tokens/test.jsonl"
all_test_files_path = "/vol/tmp/goldejon/ner/eval_data/*"

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
    # run_single_eval()
    run_complete_eval()

def run_single_eval():
    data_files = {}
    data_files["test"] = test_file
    dataset = load_dataset('json', data_files=data_files)
    run_eval(dataset)

def run_complete_eval():
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        model_name = pretrained_model_name_or_path.split("/")[-2]
        for eval_dataset in glob.glob(all_test_files_path):
            for language_dataset in glob.glob(eval_dataset + "/*"):
                dataset = DatasetDict.load_from_disk(language_dataset)

                eval_split = "test" if "test" in dataset else "dev"
                test_split = dataset[eval_split]

                max_samples = MAX_EVAL_SAMPLES_PER_DATASET[language_dataset.split("/")[-2]]
                if max_samples == -1 or max_samples > len(test_split):
                    test_split = test_split.shuffle(seed=42)
                else:
                    test_split = test_split.shuffle(seed=42).select(range(max_samples))
                if os.path.exists(f"results/{model_name}/{language_dataset.split("/")[-2]}"):
                    continue

                os.makedirs(f"results/{model_name}/{language_dataset.split("/")[-2]}", exist_ok=True)
                result_save_path = f"results/{model_name}/{language_dataset.split("/")[-2]}/{language_dataset.split("/")[-1]}.json"
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
        model = ContrastiveBiEncoderModel(config=config).to("cuda")
    else:
        model = CompressedBiEncoderModel(config=config).to("cuda")
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
        test_collator = EvalCollatorContrastiveBiEncoder(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format="tokens",
            loss_masking="subwords"
        )
    else:
        test_collator = EvalCollatorCompressedBiEncoder(
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

