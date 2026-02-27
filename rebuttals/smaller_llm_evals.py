"""Minimal script to evaluate Qwen3 models on NER using Hugging Face transformers."""

import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config (adjust as needed) ---
# List of (model_name, output_prefix) for output filenames
MODELS = [
    ("Qwen/Qwen3-0.6B", "qwen06"),
    ("Qwen/Qwen3-4B", "qwen4b"),
    # ('Qwen/Qwen3-30B-A3B-FP8', 'qwen30b'),
]
EVAL_BASE = "/vol/tmp/goldejon/multilingual_ner/data/evaluation/multinerd"
EVAL_LANGS = ["eng", "deu", "rus"]
SAMPLE_SIZE = 1000
OUTPUT_DIR = "evals"
MAX_NEW_TOKENS = 512

NER_INSTRUCTION = """Extract all named entities from the following text. Return a Python list of tuples in the format [(entity, type)], with each entity and its type as they appear in the text, preserving their original order. If there are no entities, return an empty list []. Only use the following types: {all_labels}.

Text: {text}

Entities:"""


def format_ner_prompt(text: str, all_labels: list[str]) -> str:
    return NER_INSTRUCTION.format(text=text, all_labels=all_labels)


def parse_ner_output(output: str) -> list[tuple[str, str]]:
    """Parse model output to list of (entity_text, label). Handles Python list-of-tuples string output."""
    entities = []
    output = output.strip()
    if not output or output.lower() == "none":
        return entities
    # Try parsing as Python list of tuples
    try:
        parsed = eval(output, {"__builtins__": None}, {})
        if isinstance(parsed, list):
            for tup in parsed:
                if (
                    isinstance(tup, tuple)
                    and len(tup) == 2
                    and isinstance(tup[0], str)
                    and isinstance(tup[1], str)
                ):
                    entities.append((tup[0], tup[1].lower()))
            if entities:
                return entities
    except Exception:
        pass
    # Fallback: parse line by line as "(entity, label)" or "entity [type]"
    for line in output.split("\n"):
        line = line.strip()
        # Try tuple format: ('entity', 'label')
        tuple_match = re.match(r"^\(?['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)?$", line)
        if tuple_match:
            entity_text, label = tuple_match.groups()
            if entity_text:
                entities.append((entity_text, label.lower()))
            continue
        # Try original bracket format
        match = re.match(r"^(.+?)\s*\[(\w+)\]$", line)
        if match:
            entity_text, label = match.groups()
            entity_text = entity_text.strip()
            if entity_text:
                entities.append((entity_text, label.lower()))
    return entities


def gold_from_sample(sample: dict) -> list[tuple[str, str]]:
    """Extract gold (entity_text, label) from char_spans."""
    text = sample["text"]
    spans = sample.get("spans_char", sample.get("spans_tokens", []))
    get_label = lambda s: (s.get("tag") or s.get("label", "")).lower()
    if "spans_char" in sample and spans:
        return [(text[s["start"] : s["end"]], get_label(s)) for s in spans]
    if spans and "spans_tokens" in sample:
        tokens = sample["tokens"]
        return [(" ".join(tokens[s["start"] : s["end"]]), get_label(s)) for s in spans]
    return []


def compute_metrics(gold_list: list, pred_list: list) -> tuple[float, float, float]:
    """Micro precision, recall, F1 for entity-level matching."""
    from collections import Counter

    gold_cnt = Counter(gold_list)
    pred_cnt = Counter(pred_list)
    tp = sum((gold_cnt & pred_cnt).values())
    pred_total = sum(pred_cnt.values())
    gold_total = sum(gold_cnt.values())
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run_eval_for_lang(model, tokenizer, device, lang: str, model_prefix: str):
    eval_path = f"{EVAL_BASE}/{lang}"
    output_file = f"{OUTPUT_DIR}/{model_prefix}_{lang}.jsonl"

    dataset = DatasetDict.load_from_disk(eval_path)
    split = "test" if "test" in dataset else "dev"
    dataset = dataset[split]
    all_labels = list(set([span["tag"] for sample in dataset for span in sample["spans_char"]]))
    dataset = dataset.shuffle(seed=42).select(range(min(SAMPLE_SIZE, len(dataset))))

    predictions = []
    sample_times_sec = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for i, sample in enumerate(tqdm(dataset, desc=f"{model_prefix} {lang}")):
        text = sample["text"]
        prompt = format_ner_prompt(text, all_labels)
        messages = [{"role": "user", "content": prompt}]
        model_input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer(
            [model_input_text],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Time the forward pass (GPU-synchronized when available for accuracy, same as eval_cross_encoder)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sample_times_sec.append(time.perf_counter() - t0)

        output_ids = generated_ids[0][model_inputs.input_ids.shape[1] :].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        pred_entities = parse_ner_output(output_text)
        gold_entities = gold_from_sample(sample)

        predictions.append({
            "id": sample.get("id", i),
            "text": text,
            "gold": gold_entities,
            "pred": pred_entities,
            "raw_output": output_text,
        })

    # Benchmark: VRAM, latency, FLOPs
    benchmark = {}
    if device == "cuda":
        benchmark["cuda_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    if sample_times_sec:
        n_samples = len(sample_times_sec)
        total_sec = sum(sample_times_sec)
        benchmark["n_samples"] = n_samples
        benchmark["total_time_sec"] = total_sec
        benchmark["latency_ms_per_example"] = (total_sec / n_samples) * 1000
    # FLOPs: profile a single forward pass
    try:
        with torch.no_grad():
            dummy_input = tokenizer("Hello", return_tensors="pt").to(model.device)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(**dummy_input)
            total_flops = sum(e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops)
            benchmark["flops_per_forward"] = total_flops if total_flops else None
    except Exception:
        benchmark["flops_per_forward"] = None

    # Compute metrics
    all_gold = []
    all_pred = []
    for p in predictions:
        all_gold.extend(p["gold"])
        all_pred.extend(p["pred"])

    precision, recall, f1 = compute_metrics(all_gold, all_pred)
    print(f"[{model_prefix} {lang}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if benchmark:
        print("-" * 40)
        if "cuda_memory_allocated_mb" in benchmark:
            print(f"VRAM: {benchmark['cuda_memory_allocated_mb']:.2f} MB")
        if "total_time_sec" in benchmark:
            print(f"Total time: {benchmark['total_time_sec']:.2f} s ({benchmark.get('n_samples', '?')} samples)")
        if "latency_ms_per_example" in benchmark:
            print(f"Latency: {benchmark['latency_ms_per_example']:.2f} ms/example")
        if benchmark.get("flops_per_forward") is not None:
            print(f"FLOPs: {benchmark['flops_per_forward'] / 1e9:.2f} G")
        else:
            print("FLOPs: N/A (profiler failed or unsupported)")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {output_file}")

    metrics_path = Path(output_file).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": model_prefix, "lang": lang, "precision": precision, "recall": recall, "f1": f1, "benchmark": benchmark}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model_prefix in MODELS:
        print(f"\n{'='*50}\nLoading {model_name}\n{'='*50}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

        for lang in EVAL_LANGS:
            print(f"\n{'-'*50}\n{model_prefix} - {lang}\n{'-'*50}")
            run_eval_for_lang(model, tokenizer, device, lang, model_prefix)

        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
