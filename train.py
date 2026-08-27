import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, HfArgumentParser, MT5EncoderModel, get_scheduler

from src.args import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from src.collator import (
    EvalCollatorBiEncoder,
    EvalCollatorContrastiveBiEncoder,
    EvalCollatorContrastiveCrossEncoder,
    EvalCollatorCrossEncoder,
    TrainCollatorBiEncoder,
    TrainCollatorContrastiveBiEncoder,
    TrainCollatorContrastiveCrossEncoder,
    TrainCollatorCrossEncoder,
)
from src.config import SpanModelConfig, is_bi_encoder, is_contrastive
from src.logger import setup_logger, silence_transformers_warnings
from src.model import (
    OtterBiEncoderModel,
    OtterContrastiveBiEncoderModel,
    OtterContrastiveCrossEncoderModel,
    OtterCrossEncoderModel,
)
from src.trainer import evaluate, train

# Registry: architecture name -> (model class, train collator class, eval collator class)
ARCHITECTURE_REGISTRY = {
    "bi_encoder": (OtterBiEncoderModel, TrainCollatorBiEncoder, EvalCollatorBiEncoder),
    "cross_encoder": (OtterCrossEncoderModel, TrainCollatorCrossEncoder, EvalCollatorCrossEncoder),
    "contrastive_bi_encoder": (
        OtterContrastiveBiEncoderModel,
        TrainCollatorContrastiveBiEncoder,
        EvalCollatorContrastiveBiEncoder,
    ),
    "contrastive_cross_encoder": (
        OtterContrastiveCrossEncoderModel,
        TrainCollatorContrastiveCrossEncoder,
        EvalCollatorContrastiveCrossEncoder,
    ),
}


def build_config(model_args, data_args) -> SpanModelConfig:
    return SpanModelConfig(
        architecture=model_args.architecture,
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
        prediction_threshold=model_args.prediction_threshold,
        dice_smooth=model_args.dice_smooth,
        dice_weight=model_args.dice_weight,
        focal_weight=model_args.focal_weight,
    )


def setup_model_and_tokenizers(model_args, data_args, architecture, model_cls, logger, accelerator):
    bi_encoder = is_bi_encoder(architecture)
    contrastive = is_contrastive(architecture)

    if model_args.model_checkpoint is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading model from checkpoint: {model_args.model_checkpoint}")
        config = SpanModelConfig.from_pretrained(model_args.model_checkpoint)
        config.max_span_length = data_args.max_span_length
        model = model_cls.from_pretrained(model_args.model_checkpoint)
        model.config = config

        if bi_encoder:
            token_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
            type_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
            if "mt5" in config.token_encoder:
                model.token_encoder = MT5EncoderModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            else:
                model.token_encoder = AutoModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            if "mt5" in config.type_encoder:
                model.type_encoder = MT5EncoderModel.from_pretrained(
                    config.type_encoder,
                    config=model.type_encoder.config,
                )
            else:
                model.type_encoder = AutoModel.from_pretrained(
                    config.type_encoder,
                    config=model.type_encoder.config,
                )
            tokenizers = (token_tokenizer, type_tokenizer)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_checkpoint)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
                tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
                if contrastive:
                    tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)
            tokenizers = tokenizer
    else:
        config = build_config(model_args, data_args)
        model = model_cls(config=config)

        if bi_encoder:
            token_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
            type_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
            if "mt5" in config.token_encoder:
                model.token_encoder = MT5EncoderModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            else:
                model.token_encoder = AutoModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            if "mt5" in config.type_encoder:
                model.type_encoder = MT5EncoderModel.from_pretrained(
                    config.type_encoder,
                    config=model.type_encoder.config,
                )
            else:
                model.type_encoder = AutoModel.from_pretrained(
                    config.type_encoder,
                    config=model.type_encoder.config,
                )
            tokenizers = (token_tokenizer, type_tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
            tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
            if contrastive:
                tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)
            if "mt5" in config.token_encoder:
                model.token_encoder = MT5EncoderModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            else:
                model.token_encoder = AutoModel.from_pretrained(
                    config.token_encoder,
                    config=model.token_encoder.config,
                )
            model.token_encoder.resize_token_embeddings(len(tokenizer))
            tokenizers = tokenizer

    return model, config, tokenizers


def build_train_collator(architecture, train_collator_cls, tokenizers, data_args, model_args):
    bi_encoder = is_bi_encoder(architecture)
    contrastive = is_contrastive(architecture)

    if bi_encoder:
        token_tokenizer, type_tokenizer = tokenizers
        return train_collator_cls(
            token_tokenizer,
            type_tokenizer,
            max_seq_length=data_args.max_seq_length,
            max_span_length=data_args.max_span_length,
            format=data_args.annotation_format,
            loss_masking=data_args.loss_masking,
        )
    else:
        kwargs = {
            "max_seq_length": data_args.max_seq_length,
            "max_span_length": data_args.max_span_length,
            "format": data_args.annotation_format,
            "loss_masking": data_args.loss_masking,
        }
        if contrastive:
            kwargs["prediction_threshold"] = model_args.contrastive_threshold_token
        return train_collator_cls(tokenizers, **kwargs)


def build_eval_collator(
    architecture, eval_collator_cls, tokenizers, label2id, data_args, model_args
):
    bi_encoder = is_bi_encoder(architecture)
    contrastive = is_contrastive(architecture)

    if bi_encoder:
        token_tokenizer, type_tokenizer = tokenizers
        labels = list(label2id.keys())
        type_encodings = type_tokenizer(
            labels,
            truncation=True,
            max_length=64,
            padding="longest" if len(labels) <= 1000 else "max_length",
            return_tensors="pt",
        )
        return eval_collator_cls(
            token_tokenizer,
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=data_args.max_seq_length,
            max_span_length=data_args.max_span_length,
            format=data_args.annotation_format,
            loss_masking=data_args.loss_masking,
        )
    else:
        kwargs = {
            "label2id": label2id,
            "max_seq_length": data_args.max_seq_length,
            "max_span_length": data_args.max_span_length,
            "format": data_args.annotation_format,
            "loss_masking": data_args.loss_masking,
        }
        if contrastive:
            kwargs["prediction_threshold"] = model_args.contrastive_threshold_token
        return eval_collator_cls(tokenizers, **kwargs)


def build_optimizer(model, training_args, architecture):
    bi_encoder = is_bi_encoder(architecture)

    if bi_encoder and (
        training_args.type_encoder_learning_rate is not None
        or training_args.linear_layers_learning_rate is not None
    ):
        token_encoder_params = list(model.token_encoder.parameters())
        type_encoder_params = list(model.type_encoder.parameters())
        linear_params = [
            p
            for name, p in model.named_parameters()
            if not name.startswith("token_encoder.") and not name.startswith("type_encoder.")
        ]
        param_groups = [
            {"params": token_encoder_params, "lr": training_args.learning_rate},
            {
                "params": type_encoder_params,
                "lr": training_args.type_encoder_learning_rate or training_args.learning_rate,
            },
            {
                "params": linear_params,
                "lr": training_args.linear_layers_learning_rate or training_args.learning_rate,
            },
        ]
        return torch.optim.AdamW(param_groups)

    if not bi_encoder and training_args.linear_layers_learning_rate is not None:
        token_encoder_params = list(model.token_encoder.parameters())
        linear_params = [
            p for name, p in model.named_parameters() if not name.startswith("token_encoder.")
        ]
        param_groups = [
            {"params": token_encoder_params, "lr": training_args.learning_rate},
            {"params": linear_params, "lr": training_args.linear_layers_learning_rate},
        ]
        return torch.optim.AdamW(param_groups)

    return torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)


def get_labels(dataset_split, data_args):
    span_key = "token_spans" if data_args.annotation_format == "tokens" else "char_spans"
    labels = list({span["label"] for sample in dataset_split for span in sample[span_key]})
    return {label: idx for idx, label in enumerate(labels)}


def main():
    silence_transformers_warnings()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[-1])
    )

    architecture = model_args.architecture
    if architecture is None:
        raise ValueError(
            "'architecture' must be set in the config JSON. "
            f"Valid options: {sorted(ARCHITECTURE_REGISTRY.keys())}"
        )
    if architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Valid options: {sorted(ARCHITECTURE_REGISTRY.keys())}"
        )

    model_cls, train_collator_cls, eval_collator_cls = ARCHITECTURE_REGISTRY[architecture]
    bi_encoder = is_bi_encoder(architecture)

    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(training_args.seed)

    find_unused = bi_encoder  # bi-encoder has unused params (type encoder during some steps)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)

    accelerator = Accelerator(
        mixed_precision="bf16" if getattr(training_args, "bf16", False) else "no",
        gradient_accumulation_steps=getattr(training_args, "gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp_kwargs],
    )

    best_ckpt = Path(output_dir) / "best_checkpoint"
    if best_ckpt.exists():
        if accelerator.is_main_process:
            logger = setup_logger(training_args.output_dir, is_main_process=True)
            logger.info(f"Best checkpoint already exists at {best_ckpt}. Exiting.")
        sys.exit(0)

    logger = setup_logger(training_args.output_dir, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
            f"n_gpu: {training_args.n_gpu}, "
            f"distributed training: {bool(training_args.local_rank != -1)}, "
            f"16-bits training: {training_args.bf16}"
        )

    dataset = {}
    if data_args.train_file is not None:
        dataset["train"] = load_dataset("json", data_files={"train": data_args.train_file})["train"]
    if data_args.validation_file is not None:
        dataset["validation"] = load_dataset(
            "json", data_files={"validation": data_args.validation_file}
        )["validation"]
    if data_args.test_file is not None:
        dataset["test"] = load_dataset("json", data_files={"test": data_args.test_file})["test"]

    model, config, tokenizers = setup_model_and_tokenizers(
        model_args, data_args, architecture, model_cls, logger, accelerator
    )

    train_collator = build_train_collator(
        architecture, train_collator_cls, tokenizers, data_args, model_args
    )

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train file.")
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=0,
        )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation file.")
        val_label2id = get_labels(dataset["validation"], data_args)
        eval_collator = build_eval_collator(
            architecture, eval_collator_cls, tokenizers, val_label2id, data_args, model_args
        )
        eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            num_workers=0,
        )

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test file.")
        test_label2id = get_labels(dataset["test"], data_args)
        test_collator = build_eval_collator(
            architecture, eval_collator_cls, tokenizers, test_label2id, data_args, model_args
        )
        test_dataloader = DataLoader(
            dataset["test"],
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=test_collator,
            num_workers=0,
        )

    optimizer = build_optimizer(model, training_args, architecture)
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
    )

    if training_args.do_train:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if training_args.do_eval:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if training_args.do_predict:
        test_dataloader = accelerator.prepare(test_dataloader)

    best_checkpoint_path = None
    best_f1 = 0.0
    final_step = 0

    # Cross-encoders save the tokenizer so it can be restored exactly (includes added tokens).
    # Bi-encoders reconstruct tokenizers from config.token_encoder / config.type_encoder names.
    save_tokenizer = tokenizers if not bi_encoder else None

    if training_args.do_train:
        config.save_pretrained(Path(training_args.output_dir))
        final_step, best_checkpoint_path, best_f1 = train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader if training_args.do_eval else None,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            args=training_args,
            tokenizer=save_tokenizer,
        )

    if training_args.do_predict:
        if accelerator.is_main_process:
            logger.info(f"\nTraining complete! Completed {final_step} steps.")

        if best_checkpoint_path is not None and training_args.do_eval:
            if accelerator.is_main_process:
                logger.info(f"\nLoading best model from checkpoint: {best_checkpoint_path}")
                logger.info(f"Best validation F1: {best_f1:.4f}")
            best_model = model_cls.from_pretrained(str(best_checkpoint_path))
            best_model.eval()
            best_model = accelerator.prepare(best_model)
            model = best_model
        else:
            if accelerator.is_main_process:
                logger.info("Using latest model for evaluation (no validation was performed).")
            model.eval()

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

            test_results_path = Path(training_args.output_dir) / "test_results.json"
            with open(test_results_path, "w") as f:
                json.dump(
                    {
                        "test_metrics": test_metrics,
                        "config": config.to_dict(),
                        "final_step": final_step,
                        "best_checkpoint_path": str(best_checkpoint_path)
                        if best_checkpoint_path
                        else None,
                        "best_validation_f1": best_f1,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"\nTest results saved to {test_results_path}")

    if accelerator.num_processes > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
