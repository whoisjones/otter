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
from transformers import (AutoTokenizer, get_scheduler, HfArgumentParser)
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import CompressedBiEncoderModel 
from src.config import SpanModelConfig
from src.collator import TrainCollatorCompressedBiEncoder, EvalCollatorCompressedBiEncoder
from src.trainer import train, evaluate
from src.logger import setup_logger
from src.args import ModelArguments, DataTrainingArguments, CustomTrainingArguments

transformers.logging.set_verbosity_error()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    torch.manual_seed(training_args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision="bf16" if getattr(training_args, 'bf16', False) else "no",
        gradient_accumulation_steps=getattr(training_args, 'gradient_accumulation_steps', 1),
        kwargs_handlers=[ddp_kwargs]
    )
    
    logger = setup_logger(training_args.output_dir, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
        )
    
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    dataset = load_dataset('json', data_files=data_files)

    # Load model from checkpoint if provided, otherwise initialize from scratch
    if model_args.model_checkpoint is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading model from checkpoint: {model_args.model_checkpoint}")
        # Load config from checkpoint
        config = SpanModelConfig.from_pretrained(model_args.model_checkpoint)
        # Override max_span_length from data_args as it's data-dependent
        config.max_span_length = data_args.max_span_length
        # Load model from checkpoint
        model = CompressedBiEncoderModel.from_pretrained(model_args.model_checkpoint)
        # Update model config
        model.config = config
    else:
        # Validate that token_encoder and type_encoder are provided
        if model_args.token_encoder is None or model_args.type_encoder is None:
            raise ValueError(
                "Either 'model_checkpoint' must be provided, or both 'token_encoder' and 'type_encoder' must be provided."
            )
        config = SpanModelConfig(
            token_encoder=model_args.token_encoder,
            type_encoder=model_args.type_encoder,
            loss_fn=model_args.loss_fn,
            focal_alpha=model_args.focal_alpha,
            focal_gamma=model_args.focal_gamma,
            max_span_length=data_args.max_span_length,
            dropout=model_args.dropout,
            linear_hidden_size=model_args.linear_hidden_size,
            span_width_embedding_size=model_args.span_width_embedding_size,
            init_temperature=model_args.init_temperature,
            start_loss_weight=model_args.start_loss_weight,
            end_loss_weight=model_args.end_loss_weight,
            span_loss_weight=model_args.span_loss_weight,
            bce_start_pos_weight=model_args.bce_start_pos_weight,
            bce_end_pos_weight=model_args.bce_end_pos_weight,
            bce_span_pos_weight=model_args.bce_span_pos_weight,
            contrastive_threshold_loss_weight=model_args.contrastive_threshold_loss_weight,
            contrastive_span_loss_weight=model_args.contrastive_span_loss_weight,
            contrastive_tau=model_args.contrastive_tau,
            type_encoder_pooling=model_args.type_encoder_pooling,
            prediction_threshold=model_args.prediction_threshold
        )
        model = CompressedBiEncoderModel(config=config)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
    train_collator = TrainCollatorCompressedBiEncoder(
        token_encoder_tokenizer, 
        type_encoder_tokenizer, 
        max_seq_length=data_args.max_seq_length, 
        format=data_args.annotation_format,
        loss_masking=data_args.loss_masking
    )

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train file.")
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=train_collator,
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
        eval_collator = EvalCollatorCompressedBiEncoder(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=data_args.max_seq_length, 
            format=data_args.annotation_format,
            loss_masking=data_args.loss_masking
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
        test_collator = EvalCollatorCompressedBiEncoder(
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs
    )
    
    # Prepare model, optimizer, and dataloaders with accelerator
    if training_args.do_train:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if training_args.do_eval:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if training_args.do_predict:
        test_dataloader = accelerator.prepare(test_dataloader)
    
    best_checkpoint_path = None
    best_f1 = 0.0
    final_step = 0
    
    if training_args.do_train:
        config.save_pretrained(Path(training_args.output_dir))
        final_step, best_checkpoint_path, best_f1 = train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader if training_args.do_eval else None,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            args=training_args
        )

    if training_args.do_predict:
        if accelerator.is_main_process:
            logger.info(f"\nTraining complete! Completed {final_step} steps.")
        
        # Load best model for evaluation
        if best_checkpoint_path is not None and training_args.do_eval:
            if accelerator.is_main_process:
                logger.info(f"\nLoading best model from checkpoint: {best_checkpoint_path}")
                logger.info(f"Best validation F1: {best_f1:.4f}")
            # Load the best model
            best_model = CompressedBiEncoderModel.from_pretrained(str(best_checkpoint_path))
            best_model.eval()
            best_model = accelerator.prepare(best_model)
            model = best_model
        else:
            if accelerator.is_main_process:
                logger.info("Using latest model for evaluation (no validation was performed during training).")
            model.eval()
        
        # Final evaluation on test set
        if accelerator.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info("Final Test Set Evaluation")
            logger.info("=" * 60)
        test_metrics = evaluate(model, test_dataloader, accelerator)
    
        if accelerator.is_main_process:
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Precision: {test_metrics['micro']['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['micro']['recall']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['micro']['f1']:.4f}")
            logger.info("=" * 60)
        
        # Save test results to file (only on main process)
        if accelerator.is_main_process:
            test_results_path = Path(training_args.output_dir) / "test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump({
                    "test_metrics": test_metrics,
                    "config": config.to_dict(),
                    "final_step": final_step,
                    "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                    "best_validation_f1": best_f1
                }, f, indent=2)
            logger.info(f"\nTest results saved to {test_results_path}")

    if accelerator.num_processes > 1:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

