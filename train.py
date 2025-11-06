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
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup, HfArgumentParser, TrainingArguments)
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
from src.collator import InBatchDataCollator, AllLabelsDataCollator
from src.trainer import train, evaluate
from src.logger import setup_logger
from src.args import ModelArguments, DataTrainingArguments

transformers.logging.set_verbosity_error()


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
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    dataset = load_dataset('json', data_files=data_files)

    config = SpanModelConfig(
        token_encoder=model_args.token_encoder,
        type_encoder=model_args.type_encoder,
        max_span_length=data_args.max_span_length,
        dropout=model_args.dropout,
        linear_hidden_size=model_args.linear_hidden_size,
        init_temperature=model_args.init_temperature,
        start_loss_weight=model_args.start_loss_weight,
        end_loss_weight=model_args.end_loss_weight,
        span_loss_weight=model_args.span_loss_weight,
        type_encoder_pooling=model_args.type_encoder_pooling,
        use_pos_weight=model_args.use_pos_weight,
        prediction_threshold=model_args.prediction_threshold
    )
    model = SpanModel(config=config)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
    in_batch_collator = InBatchDataCollator(token_encoder_tokenizer, type_encoder_tokenizer, max_seq_length=data_args.max_seq_length, format=data_args.annotation_format)

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train file.")
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=in_batch_collator,
            num_workers=0
        )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation file.")
        validation_labels = list(set([span["label"] for sample in dataset["validation"] for span in sample["token_spans" if data_args.annotation_format == "tokens" else "char_spans"]]))
        label2id = {label: idx for idx, label in enumerate(validation_labels)}
        type_encodings = type_encoder_tokenizer(
            list(label2id.keys()),
            truncation=True,
            max_length=64,
            padding="longest" if len(validation_labels) <= 1000 else "max_length",
            return_tensors="pt"
        )
        eval_collator = AllLabelsDataCollator(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=data_args.max_seq_length, 
            format=data_args.annotation_format
        )
        eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            num_workers=0
        )
    
    if training_args.do_predict:
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
            format=data_args.annotation_format
        )
        test_dataloader = DataLoader(
            dataset["test"],
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=test_collator,
            num_workers=0
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )
    
    # Prepare model, optimizer, and dataloaders with accelerator
    if training_args.do_train:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if training_args.do_eval:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if training_args.do_predict:
        test_dataloader = accelerator.prepare(test_dataloader)
    
    if training_args.do_train:
        config.save_pretrained(Path(training_args.output_dir))
        final_step = train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader if training_args.do_eval else None,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            args=training_args
        )

    if training_args.do_predict:
        logger.info(f"\nTraining complete! Completed {final_step} steps.")
        
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
        
        # Save test results to file
        test_results_path = Path(training_args.output_dir) / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump({
                "test_metrics": test_metrics,
                "config": config.to_dict(),
                "final_step": final_step
            }, f, indent=2)
        logger.info(f"\nTest results saved to {test_results_path}")


if __name__ == "__main__":
    main()

