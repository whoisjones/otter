import argparse
import ast
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from src.hub.model_card import EXAMPLE_LABELS, EXAMPLE_TEXT
from src.hub.model_card import render as render_model_card

ROOT = Path(__file__).parent
HUB_SOURCES = ROOT / "src" / "hub"

# Copied verbatim from src/ into every published repo.
MIRRORED = {
    "masks.py": ROOT / "src" / "collator" / "masks.py",
    "loss.py": ROOT / "src" / "loss.py",
    "metrics.py": ROOT / "src" / "metrics.py",
}

# Concatenated into a single collate_fn.py, since the Hub repos are flat.
COLLATORS = [
    ROOT / "src" / "collator" / "train_collator_biencoder.py",
    ROOT / "src" / "collator" / "eval_collator_biencoder.py",
    ROOT / "src" / "collator" / "train_collator_crossencoder.py",
    ROOT / "src" / "collator" / "eval_collator_crossencoder.py",
]

REPOS = {
    "otter-bi-mmbert": {
        "source": "whoisjones/otter-bi-mmbert",
        "architecture": "bi_encoder",
        "token_encoder": "jhu-clsp/mmBERT-base",
        "type_encoder": "google-bert/bert-base-multilingual-uncased",
        "max_seq_length": 1024,
    },
    "otter-cross-mmbert": {
        "source": "whoisjones/otter-cross-mmbert",
        "architecture": "cross_encoder",
        "token_encoder": "jhu-clsp/mmBERT-base",
        "type_encoder": "none",
        "max_seq_length": 1024,
    },
    "otter-bi-rembert": {
        "source": "whoisjones/otter-bi-rembert",
        "architecture": "bi_encoder",
        "token_encoder": "google/rembert",
        "type_encoder": "google-bert/bert-base-multilingual-uncased",
        "max_seq_length": 512,
    },
    "otter-cross-rembert": {
        "source": "whoisjones/otter-cross-rembert",
        "architecture": "cross_encoder",
        "token_encoder": "google/rembert",
        "type_encoder": "none",
        "max_seq_length": 512,
    },
}

MODEL_CLASS = {"bi_encoder": "OtterBiEncoderModel", "cross_encoder": "OtterCrossEncoderModel"}
CONFIG_CLASS = {"bi_encoder": "OtterBiEncoderConfig", "cross_encoder": "OtterCrossEncoderConfig"}
MODEL_TYPE = {"bi_encoder": "otter-bi-encoder", "cross_encoder": "otter-cross-encoder"}

# Thresholds from the macro-F1 calibration sweep over the evaluation suite
# (results/calibration_summary_language_splits_{bi,ce}.txt). The values that shipped in
# the original config.json were inconsistent across checkpoints -- 0.5, 0.5, 0.1, 0.3 --
# and did not all match the calibration.
THRESHOLD = {"bi_encoder": 0.2, "cross_encoder": 0.5}

# Config keys that describe the trained model. Anything else in the old config.json
# (bookkeeping written by an older transformers) is dropped.
CARRIED_KEYS = [
    "loss_fn",
    "max_span_length",
    "linear_hidden_size",
    "span_width_embedding_size",
    "dropout",
    "init_temperature",
    "prediction_threshold",
    "start_loss_weight",
    "end_loss_weight",
    "span_loss_weight",
    "bce_start_pos_weight",
    "bce_end_pos_weight",
    "bce_span_pos_weight",
    "focal_alpha",
    "focal_gamma",
    "contrastive_threshold_loss_weight",
    "contrastive_span_loss_weight",
    "contrastive_tau",
    "type_encoder_pooling",
]

# Files earlier uploads left behind that the current layout replaces.
STALE = [
    "configuration_biencoder.py",
    "modeling_biencoder.py",
    "configuration_crossencoder.py",
    "modeling_crossencoder.py",
]


def build_collate_fn():
    header = [
        '"""Collators for fine-tuning Otter models.\n\n'
        'Generated from src/collator/ -- see https://github.com/whoisjones/otter.\n"""\n'
    ]
    seen, bodies = set(), []
    for path in COLLATORS:
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source)
        import_lines = set()
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            import_lines.update(range(node.lineno, node.end_lineno + 1))
            statement = ast.unparse(node)
            if statement not in seen:
                seen.add(statement)
                header.append(statement)
        body = [line for i, line in enumerate(lines, start=1) if i not in import_lines]
        bodies.append("\n".join(body).strip("\n"))
    return "\n".join(header) + "\n\n\n" + "\n\n\n".join(bodies) + "\n"


def _fetch(repo, filename, dest_dir):
    try:
        path = hf_hub_download(repo, filename)
    except Exception:
        return None
    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, target)
    return target


def _encoder_config_dict(out_dir, filename, fallback_name):
    path = out_dir / filename
    config = AutoConfig.from_pretrained(str(path) if path.exists() else fallback_name)
    data = config.to_dict()
    for key in (
        "transformers_version",
        "_attn_implementation_autoset",
        "architectures",
        "_name_or_path",
        "torch_dtype",
    ):
        data.pop(key, None)
    return data


def _stage_tokenizers(spec, out_dir):
    if spec["architecture"] == "bi_encoder":
        for subfolder, fallback in (
            ("token_tokenizer", spec["token_encoder"]),
            ("type_tokenizer", spec["type_encoder"]),
        ):
            if (out_dir / subfolder).exists() and any((out_dir / subfolder).iterdir()):
                continue
            AutoTokenizer.from_pretrained(fallback).save_pretrained(str(out_dir / subfolder))
            print(f"    + {subfolder}/ (from {fallback})")
    elif not (out_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(spec["token_encoder"])
        tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"    + tokenizer (from {spec['token_encoder']} + [LABEL])")


def _run_example(out_dir):
    if not (out_dir / "model.safetensors").exists():
        print("    (no local weights -- model card omits sample output)")
        return None
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(out_dir), trust_remote_code=True)
    model.eval()
    lines = [
        f"{e['text']!r:25} {e['label']:15} {e['score']:.2f}"
        for e in model.predict(EXAMPLE_TEXT, labels=EXAMPLE_LABELS)
    ]
    del model
    return "\n".join(lines)


def build(name, spec, out_root, with_example=False):
    out_dir = out_root / name
    # Weights are never rebuilt here, and a linked-in copy is what lets --example run,
    # so carry any existing model.safetensors across the rebuild.
    weights = out_dir / "model.safetensors"
    carried = None
    if weights.is_symlink():
        carried = weights.readlink()
    elif weights.exists():
        carried = weights.read_bytes()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if isinstance(carried, Path):
        weights.symlink_to(carried)
    elif carried is not None:
        weights.write_bytes(carried)

    source = spec["source"]
    print(f"\n=== {name}  (from {source})")

    # Pull everything except the weights and the files regenerated below.
    for filename in sorted(s.rfilename for s in HfApi().model_info(source).siblings):
        if filename.endswith(".py"):
            continue
        if filename in ("model.safetensors", "config.json", "README.md", ".gitattributes"):
            continue
        _fetch(source, filename, out_dir)
    old_config = json.loads(Path(hf_hub_download(source, "config.json")).read_text())

    _stage_tokenizers(spec, out_dir)

    architecture = spec["architecture"]
    config = {key: old_config.get(key) for key in CARRIED_KEYS}
    config.update(
        {
            "model_type": MODEL_TYPE[architecture],
            "architectures": [MODEL_CLASS[architecture]],
            "auto_map": {
                "AutoConfig": f"configuration_otter.{CONFIG_CLASS[architecture]}",
                "AutoModel": f"modeling_otter.{MODEL_CLASS[architecture]}",
                "AutoModelForTokenClassification": f"modeling_otter.{MODEL_CLASS[architecture]}",
            },
            "architecture": architecture,
            "token_encoder": spec["token_encoder"],
            "type_encoder": spec["type_encoder"],
            "max_seq_length": spec["max_seq_length"],
            "prediction_threshold": THRESHOLD[architecture],
            "token_encoder_config": _encoder_config_dict(
                out_dir, "token_encoder_config.json", spec["token_encoder"]
            ),
            "dtype": "float32",
        }
    )
    if architecture == "bi_encoder":
        config["type_encoder_config"] = _encoder_config_dict(
            out_dir, "type_encoder_config.json", spec["type_encoder"]
        )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    for module in ("configuration_otter.py", "modeling_otter.py"):
        shutil.copy(HUB_SOURCES / module, out_dir / module)
    for filename, path in MIRRORED.items():
        shutil.copy(path, out_dir / filename)
    (out_dir / "collate_fn.py").write_text(build_collate_fn())

    example = _run_example(out_dir) if with_example else None
    (out_dir / "README.md").write_text(
        render_model_card(name, spec, THRESHOLD[architecture], example_output=example)
    )

    print(f"    -> {out_dir}")
    return out_dir


def push(name, spec, out_dir, api, user="whoisjones"):
    source, target = spec["source"], f"{user}/{name}"
    if source != target:
        print(f"    renaming {source} -> {target}")
        api.move_repo(from_id=source, to_id=target, repo_type="model")

    existing = {s.rfilename for s in api.model_info(target).siblings}
    api.upload_folder(
        repo_id=target,
        folder_path=str(out_dir),
        ignore_patterns=["model.safetensors"],
        commit_message="Self-contained transformers integration: predict() API, "
        "bundled tokenizers, calibrated threshold, new model card",
    )
    for filename in STALE:
        if filename in existing:
            api.delete_file(
                filename,
                repo_id=target,
                repo_type="model",
                commit_message=f"Remove superseded {filename}",
            )
            print(f"    removed stale {filename}")
    print(f"    pushed https://huggingface.co/{target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", nargs="*", choices=sorted(REPOS), default=sorted(REPOS))
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run each model to put its real output in the model card. "
        "Requires model.safetensors in the staged directory.",
    )
    parser.add_argument("--push", action="store_true", help="Publish the staged repos.")
    args = parser.parse_args()

    api = HfApi()
    for name in args.only:
        out_dir = build(name, REPOS[name], Path(args.out_dir), with_example=args.example)
        if args.push:
            push(name, REPOS[name], out_dir, api)


if __name__ == "__main__":
    main()
