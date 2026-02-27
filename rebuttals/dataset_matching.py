#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter

import torch
import numpy as np
from scipy import linalg
from datasets import DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": -1,
    "multinerd": 1000,
    "multiconer_v1": 1000,
    "multiconer_v2": 1000,
    "dynamicner": -1,
    "uner": -1,
}

# Paths
EVAL_BASE = "/vol/tmp/goldejon/multilingual_ner/data/evaluation"
FINERWEB_PATH = "/vol/tmp/goldejon/ner/data/finerweb_splitted"
EURO_GLINERX_PATH = "/vol/tmp/goldejon/multilingual_ner/data/training_jsonl/euro_glinerx"
PILENER_PATH = "/vol/tmp/goldejon/multilingual_ner/data/training_jsonl/pilener"

NON_WHITESPACE_LANGS = ["cmn", "zho", "mal", "yue", "jpn", "tha", "khm", "lao", "mya", "bod", "dzo"]


def load_eval_benchmark(benchmark: str, lang: str) -> DatasetDict:
    """Load evaluation benchmark texts. Uses train split, fallback to dev."""
    path = Path(EVAL_BASE) / benchmark / lang
    dataset = DatasetDict.load_from_disk(str(path))
    return dataset


def load_training_corpus(corpus_name: str, languages: Optional[List[str]] = None) -> DatasetDict:
    """Load training corpus texts. Returns list of text strings."""
    if corpus_name == "finerweb":
        base = Path(FINERWEB_PATH)
        if languages is not None:
            data_files = {d.split('.')[0]: f"{base}/{d}.jsonl" for d in languages if f"{d}.jsonl" in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
        else:
            data_files = {d.split('.')[0]: f"{base}/{d}" for d in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
    elif corpus_name == "euro_glinerx":
        base = Path(EURO_GLINERX_PATH)
        if languages is not None:
            data_files = {d.split('.')[0]: f"{base}/{d}.jsonl" for d in languages if f"{d}.jsonl" in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
        else:
            data_files = {d.split('.')[0]: f"{base}/{d}" for d in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
    elif corpus_name == "pilener":
        base = Path(PILENER_PATH)
        if languages is not None:
            data_files = {d.split('.')[0]: f"{base}/{d}.jsonl" for d in languages if f"{d}.jsonl" in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
        else:
            data_files = {d.split('.')[0]: f"{base}/{d}" for d in os.listdir(base)}
            dataset = load_dataset("json", data_files=data_files)
    return dataset

def _char_ngrams(s: str, n: int, whitespace_lang: bool = False) -> List[str]:
    if whitespace_lang:
        s = " ".join(s.split())
    else:
        s = s.replace(" ", "")
    if len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def _ngram_counts(sentences: List[str], n_values: Tuple[int, ...], whitespace_lang: bool = False) -> Counter:
    c = Counter()
    for s in sentences:
        for n in n_values:
            c.update(_char_ngrams(s, n, whitespace_lang))
    return c

def _js_divergence_from_counts(c1: Counter, c2: Counter, eps: float = 1e-12) -> float:
    # Jensen–Shannon divergence using log base 2, result in [0, 1]
    keys = set(c1.keys()) | set(c2.keys())
    tot1 = sum(c1.values())
    tot2 = sum(c2.values())
    if tot1 == 0 or tot2 == 0:
        return 1.0  # maximally different if one side has no ngrams

    jsd = 0.0
    for k in keys:
        p = c1.get(k, 0) / tot1
        q = c2.get(k, 0) / tot2
        m = 0.5 * (p + q)

        # KL(p||m) + KL(q||m) with safe guards
        if p > 0:
            jsd += 0.5 * p * math.log((p + eps) / (m + eps), 2)
        if q > 0:
            jsd += 0.5 * q * math.log((q + eps) / (m + eps), 2)

    # Numerically, tiny negatives can occur; clamp
    return max(0.0, min(1.0, jsd))

def lexical_analysis(training_data: DatasetDict, benchmarks_data: DatasetDict, languages: List[str]) -> dict:
    results = {}
    for lang in languages:
        for train_dataset_name, train_dataset in training_data.items():
            if lang not in train_dataset:
                continue
            benchmark_dataset = benchmarks_data[f"multinerd/{lang}"]
            benchmark_dataset = benchmark_dataset['test' if 'test' in benchmark_dataset else 'dev']
            n = min(5000, len(train_dataset[lang]))
            eval_counts = _ngram_counts(list(benchmark_dataset['text']), (3, 4, 5), whitespace_lang=True if not lang == "zho" else False)
            jsd_vals: List[float] = []
            for k in range(20):
                print(f"Running {train_dataset_name}/{lang} with seed {k}")
                train_sample = train_dataset[lang].shuffle(seed=k).select(range(n))
                train_counts = _ngram_counts(list(train_sample['text']), (3, 4, 5), whitespace_lang=True if not lang == "zho" else False)
                jsd_vals.append(_js_divergence_from_counts(train_counts, eval_counts))

            mean_jsd = sum(jsd_vals) / len(jsd_vals)
            var = sum((x - mean_jsd) ** 2 for x in jsd_vals) / max(1, (len(jsd_vals) - 1))
            std_jsd = math.sqrt(var)
            results[f"{train_dataset_name}/{lang}"] = {
                "mean_jsd": mean_jsd,
                "std_jsd": std_jsd,
            }
    return results

def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def embedding_analysis(training_data: DatasetDict, benchmarks_data: DatasetDict, languages: List[str]) -> dict:
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to("cuda")
    results = {}
    for lang in languages:
        for train_dataset_name, train_dataset in training_data.items():
            if lang not in train_dataset:
                continue
            benchmark_dataset = benchmarks_data[f"multinerd/{lang}"]
            benchmark_dataset = benchmark_dataset['test' if 'test' in benchmark_dataset else 'dev']
            n = min(5000, len(train_dataset[lang]))
            f_dists: List[float] = []
            for k in range(20):
                print(f"Running {train_dataset_name}/{lang} with seed {k}")
                train_sample = train_dataset[lang].shuffle(seed=k).select(range(n))
                emb_a = model.encode(list(train_sample['text']), batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
                emb_b = model.encode(list(benchmark_dataset['text']), batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
                mu_a, mu_b = emb_a.mean(0), emb_b.mean(0)
                cov_a, cov_b = np.cov(emb_a, rowvar=False), np.cov(emb_b, rowvar=False)
                f_dist = _frechet_distance(mu_a, cov_a, mu_b, cov_b)
                f_dists.append(f_dist)
            mean_f_dist = sum(f_dists) / len(f_dists)
            var = sum((x - mean_f_dist) ** 2 for x in f_dists) / max(1, (len(f_dists) - 1))
            std_f_dist = math.sqrt(var)
            results[f"{train_dataset_name}/{lang}"] = {
                "mean_f_dist": mean_f_dist,
                "std_f_dist": std_f_dist,
            }
    return results

def run_analysis(
    benchmarks: Optional[List[str]] = None,
    corpora: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> dict:
    if benchmarks is None:
        benchmarks = [
            d for d in os.listdir(EVAL_BASE)
            if os.path.isdir(os.path.join(EVAL_BASE, d))
        ]
    if corpora is None:
        corpora = ["finerweb", "euro_glinerx", "pilener"]

    benchmarks_data = {}
    for benchmark in tqdm(sorted(benchmarks), desc="Loading eval benchmarks", leave=False):
        bench_path = Path(EVAL_BASE) / benchmark
        if not bench_path.is_dir():
            continue
        langs = [d for d in os.listdir(bench_path) if (bench_path / d).is_dir()]
        for lang in tqdm(sorted(langs), desc="Loading eval benchmarks", leave=False):
            if languages is not None and lang not in languages:
                continue
            dataset = load_eval_benchmark(benchmark, lang)
            key = f"{benchmark}/{lang}"
            benchmarks_data[key] = dataset

    training_data = {}
    for corpus in corpora:
        if corpus not in training_data:
            dataset = load_training_corpus(corpus, languages)
            training_data[corpus] = dataset

    lexical_results = lexical_analysis(training_data, benchmarks_data, languages)
    embedding_results = embedding_analysis(training_data, benchmarks_data, languages)

    with open(output_path, "w") as f:
        json.dump({
            "lexical_results": lexical_results,
            "embedding_results": embedding_results,
        }, f, indent=2)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check overlap between evaluation benchmarks and training corpora"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Evaluation benchmarks to check (default: all)",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=["finerweb", "euro_glinerx", "pilener"],
        default=None,
        help="Training corpora to compare (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to check (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="JSON output path (default: rebuttal_overlap_results.json in working dir)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "rebuttal_overlap_results.json")

    run_analysis(
        benchmarks=args.benchmarks,
        corpora=args.corpora,
        languages=args.languages,
        output_path=output_path,
    )

if __name__ == "__main__":
    main()
