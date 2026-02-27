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

from src.model import OtterBiEncoderModel, OtterContrastiveBiEncoderModel 
from src.config import SpanModelConfig
from src.collator import EvalCollatorBiEncoder, EvalCollatorContrastiveBiEncoder
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()


def run_eval(
    pretrained_model_name_or_path, 
    dataset, 
    result_save_path, 
    prediction_threshold = None, 
    evaluation_format = "text",
    identifier = None
    ):
    logger = setup_logger('eval_bi')
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
        model = OtterContrastiveBiEncoderModel(config=config).to("cuda")
    else:
        model = OtterBiEncoderModel(config=config).to("cuda")
    model = model.from_pretrained(pretrained_model_name_or_path)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)

    test_labels = list(set([span["label"] for sample in dataset for span in sample["token_spans"]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}

    if prediction_threshold is not None: 
        model.config.prediction_threshold = prediction_threshold

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
            format=evaluation_format,
            loss_masking="none" if evaluation_format == "text" else "subwords"
        )
    else:
        test_collator = EvalCollatorBiEncoder(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format=evaluation_format,
            loss_masking="none" if evaluation_format == "text" else "subwords"
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

    if identifier is not None:
        test_metrics['eval_from'] = identifier
        
    with open(result_save_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
        }, f, indent=2)
    logger.info(f"\nTest results saved to {result_save_path}")

def main(args):
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required when evaluating a single model")
    if args.evaluation_dataset.endswith(".jsonl"):
        test_split = {f'{args.evaluation_dataset}': load_dataset('json', data_files=args.evaluation_dataset, split="train")}
    elif os.path.exists(args.evaluation_dataset) and os.path.isdir(args.evaluation_dataset):
        dataset = DatasetDict.load_from_disk(args.evaluation_dataset)
        eval_split = "test" if "test" in dataset else "dev"
        test_split = dataset[eval_split]
        test_split = {f'{args.evaluation_dataset}': test_split}
    else:
        config_names = get_dataset_config_names(args.evaluation_dataset)
        test_splits = {}
        for config_name in config_names:
            dataset = load_dataset(args.evaluation_dataset, config_name)
            eval_split = "test" if "test" in dataset else "dev"
            test_split = dataset[eval_split]
            test_splits[f'{config_name}'] = test_split
    
    for identifier, test_split in test_splits.items():
        os.makedirs("evals", exist_ok=True)
        result_save_path = os.path.join("evals", datetime.now().strftime("eval_%Y%m%d_%H%M%S.json"))
        if len(test_split) > args.max_eval_samples:
            test_split = test_split.shuffle(seed=42).select(range(args.max_eval_samples))
        run_eval(args.pretrained_model_name_or_path, test_split, result_save_path, args.threshold, args.evaluation_format, identifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--evaluation_format", type=str, choices=["text", "tokens"], default="text")
    args = parser.parse_args()
    main(args)

