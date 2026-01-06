import os
import argparse
import glob
import json
import warnings

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

from src.model import CompressedCrossEncoderModel, ContrastiveCrossEncoderModel 
from src.config import SpanModelConfig
from src.collator import EvalCollatorCompressedCrossEncoder, EvalCollatorContrastiveCrossEncoder
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()

MODEL_DIR = "/vol/tmp2/goldejon/paper_experiments"
ABLATION_DIR = "/vol/tmp2/goldejon/loss_function_ablation"

TEST_FILES = {
    "panx": "/vol/tmp/goldejon/ner/eval_data/panx",
    "masakhaner": "/vol/tmp/goldejon/ner/eval_data/masakhaner",
    "multinerd": "/vol/tmp/goldejon/ner/eval_data/multinerd",
    "multiconer_v1": "/vol/tmp/goldejon/ner/eval_data/multiconer_v1",
    "multiconer_v2": "/vol/tmp/goldejon/ner/eval_data/multiconer_v2",
    "dynamicner": "/vol/tmp/goldejon/ner/eval_data/dynamicner",
    "uner": "/vol/tmp/goldejon/ner/eval_data/uner",
    "panx_translated": "/vol/tmp/goldejon/ner/eval_data_translated/panx",
    "masakhaner_translated": "/vol/tmp/goldejon/ner/eval_data_translated/masakhaner",
    "multinerd_translated": "/vol/tmp/goldejon/ner/eval_data_translated/multinerd",
    "multiconer_v1_translated": "/vol/tmp/goldejon/ner/eval_data_translated/multiconer_v1",
    "multiconer_v2_translated": "/vol/tmp/goldejon/ner/eval_data_translated/multiconer_v2",
    "dynamicner_translated": "/vol/tmp/goldejon/ner/eval_data_translated/dynamicner",
    "uner_translated": "/vol/tmp/goldejon/ner/eval_data_translated/uner",
}

MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": 1000,
    "multinerd": 1000,
    "multiconer_v1": 1000,
    "multiconer_v2": 1000,
    "dynamicner": 1000,
    "uner": 1000,
    "panx_translated": 1000,
    "masakhaner_translated": 1000,
    "multinerd_translated": 1000,
    "multiconer_v1_translated": 1000,
    "multiconer_v2_translated": 1000,
    "dynamicner_translated": 1000,
    "uner_translated": 1000,
}

def run_eval(dataset, pretrained_model_name_or_path, result_save_path, prediction_threshold = None):
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
        model = ContrastiveCrossEncoderModel(config=config).to("cuda")
    else:
        model = CompressedCrossEncoderModel(config=config).to("cuda")
    model = model.from_pretrained(pretrained_model_name_or_path)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path + "/tokenizer")

    test_labels = list(set([span["label"] for sample in dataset for span in sample["token_spans"]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}

    if prediction_threshold is not None:
        model.config.prediction_threshold = prediction_threshold

    if config.loss_fn == "contrastive":
        test_collator = EvalCollatorContrastiveCrossEncoder(
            token_encoder_tokenizer, 
            label2id=label2id,
            max_seq_length=512 if config.token_encoder != "jhu-clsp/mmBERT-base" else 1024, 
            format="text",
            loss_masking="none"
        )
    else:
        test_collator = EvalCollatorCompressedCrossEncoder(
            token_encoder_tokenizer, 
            label2id=label2id,
            max_seq_length=512 if config.token_encoder != "jhu-clsp/mmBERT-base" else 1024, 
            format="text",
            loss_masking="none"
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

def get_test_split(dataset, max_samples):
    eval_split = "test" if "test" in dataset else "dev"
    test_split = dataset[eval_split]
    if max_samples == -1 or max_samples > len(test_split):
        test_split = test_split.shuffle(seed=42)
    else:
        test_split = test_split.shuffle(seed=42).select(range(max_samples))
    return test_split

def run_single_eval(pretrained_model_name_or_path):
    model_name = pretrained_model_name_or_path.split("/")[-2]
    for eval_dataset_name, eval_dataset_path in TEST_FILES.items():
        for language_dataset in glob.glob(eval_dataset_path + "/*"):
            if 'translated' in language_dataset:
                continue
            dataset = DatasetDict.load_from_disk(language_dataset)
            max_samples = MAX_EVAL_SAMPLES_PER_DATASET[eval_dataset_name]
            test_split = get_test_split(dataset, max_samples)
            thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5] if not "contrastive" in model_name else ["label_token"]
            for prediction_threshold in thresholds:
                result_save_path = f"/vol/tmp/goldejon/ner/paper_results/test_runs/{model_name}/{eval_dataset_name}/{language_dataset.split("/")[-1]}_{prediction_threshold}.json"
                if os.path.exists(result_save_path):
                    continue
                os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
                run_eval(test_split, pretrained_model_name_or_path, result_save_path, prediction_threshold)

def run_complete_eval_xx():
    for pretrained_model_name_or_path in glob.glob(MODEL_DIR + "/*"):
        model_name = pretrained_model_name_or_path.split("/")[-1]
        if not model_name in ["ce_xlmr_finerweb", "ce_mmbert_finerweb"]:
            continue
        for eval_dataset_name, eval_dataset_path in TEST_FILES.items():
            for language_dataset in glob.glob(eval_dataset_path + "/*"):
                if 'translated' in language_dataset:
                    continue
                dataset = DatasetDict.load_from_disk(language_dataset)
                max_samples = MAX_EVAL_SAMPLES_PER_DATASET[eval_dataset_name]
                test_split = get_test_split(dataset, max_samples)
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
                for prediction_threshold in thresholds:
                    result_save_path = f"/vol/tmp/goldejon/ner/paper_results/test_runs/{model_name}/{eval_dataset_name}/{language_dataset.split("/")[-1]}_{prediction_threshold}.json"
                    if os.path.exists(result_save_path):
                        continue
                    print(f"Running evaluation on {language_dataset}")
                    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
                    run_eval(test_split, pretrained_model_name_or_path + "/best_checkpoint", result_save_path, prediction_threshold)

def run_complete_eval():
    for pretrained_model_name_or_path in glob.glob(ABLATION_DIR + "/*"):
        model_name = pretrained_model_name_or_path.split("/")[-1]
        if not model_name.startswith("ce_"):
            continue
        for eval_dataset_name, eval_dataset_path in TEST_FILES.items():
            for language_dataset in glob.glob(eval_dataset_path + "/*"):
                if 'translated' in language_dataset:
                    continue
                dataset = DatasetDict.load_from_disk(language_dataset)
                max_samples = MAX_EVAL_SAMPLES_PER_DATASET[eval_dataset_name]
                test_split = get_test_split(dataset, max_samples)
                thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5] if not "contrastive" in model_name else ["cls"]
                for prediction_threshold in thresholds:
                    result_save_path = f"/vol/tmp/goldejon/ner/paper_results/loss_function_ablation/{model_name}/{eval_dataset_name}/{language_dataset.split("/")[-1]}_{prediction_threshold}.json"
                    if os.path.exists(result_save_path):
                        continue
                    print(f"Running evaluation on {language_dataset}")
                    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
                    run_eval(test_split, pretrained_model_name_or_path + "/best_checkpoint", result_save_path, prediction_threshold)

def main(args):
    if args.complete_eval:
        run_complete_eval()
    else:
        if args.pretrained_model_name_or_path is None:
            raise ValueError("--pretrained_model_name_or_path is required when evaluating a single dataset")
        run_single_eval(args.pretrained_model_name_or_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=False)
    parser.add_argument("--complete_eval", action="store_true")
    args = parser.parse_args()
    main(args)

