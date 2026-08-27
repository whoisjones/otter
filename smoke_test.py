import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, BertConfig, HfArgumentParser

from src.args import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from src.config import SpanModelConfig
from src.logger import silence_transformers_warnings
from train import ARCHITECTURE_REGISTRY, build_eval_collator, build_train_collator, is_bi_encoder

BASE_ENCODER = "google-bert/bert-base-uncased"

SAMPLES = [
    {
        "text": "John Doe works at OpenAI in San Francisco.",
        "char_spans": [
            {"start": 0, "end": 8, "label": "person"},
            {"start": 18, "end": 24, "label": "organization"},
            {"start": 28, "end": 41, "label": "location"},
        ],
    },
    {
        "text": "Amazon was founded by Jeff Bezos.",
        "char_spans": [
            {"start": 0, "end": 6, "label": "organization"},
            {"start": 22, "end": 32, "label": "person"},
        ],
    },
]


def make_tiny_encoder(directory: Path) -> str:
    # A two-layer BERT keeps the dry run fast; only the tokenizer comes from the Hub.
    BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=30522,
    ).save_pretrained(directory)
    AutoTokenizer.from_pretrained(BASE_ENCODER).save_pretrained(directory)
    return str(directory)


def check_configs():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    for path in sorted(Path("configs").glob("*.json")):
        model_args, _, _ = parser.parse_json_file(json_file=str(path.resolve()))
        assert model_args.architecture in ARCHITECTURE_REGISTRY, path
        print(f"  config ok: {path} -> {model_args.architecture}")


def build_tokenizers(encoder, architecture, contrastive):
    if is_bi_encoder(architecture):
        return (
            AutoTokenizer.from_pretrained(encoder),
            AutoTokenizer.from_pretrained(encoder),
        )
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
    if contrastive:
        tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)
    return tokenizer


def check_architecture(architecture, encoder):
    contrastive = "contrastive" in architecture
    model_cls, train_collator_cls, eval_collator_cls = ARCHITECTURE_REGISTRY[architecture]

    config = SpanModelConfig(
        architecture=architecture,
        token_encoder=encoder,
        type_encoder=encoder,
        loss_fn="contrastive" if contrastive else "bce",
        max_span_length=8,
        linear_hidden_size=32,
        span_width_embedding_size=16,
        dropout=0.1,
        contrastive_tau=1.0,
    )
    model = model_cls(config=config)
    tokenizers = build_tokenizers(encoder, architecture, contrastive)
    if not is_bi_encoder(architecture):
        model.token_encoder.resize_token_embeddings(len(tokenizers))

    model_args = SimpleNamespace(contrastive_threshold_token="label_token" if contrastive else None)
    data_args = SimpleNamespace(
        max_seq_length=128,
        max_span_length=8,
        annotation_format="text",
        loss_masking="none",
    )

    train_collator = build_train_collator(
        architecture, train_collator_cls, tokenizers, data_args, model_args
    )
    batch = train_collator(SAMPLES)
    model.train()
    output = model(
        token_encoder_inputs=batch["token_encoder_inputs"],
        type_encoder_inputs=batch.get("type_encoder_inputs"),
        labels=batch["labels"],
    )
    assert torch.isfinite(output.loss), f"{architecture}: non-finite training loss"

    label2id = {"person": 0, "organization": 1, "location": 2}
    eval_collator = build_eval_collator(
        architecture, eval_collator_cls, tokenizers, label2id, data_args, model_args
    )
    batch = eval_collator(SAMPLES)
    model.eval()
    with torch.no_grad():
        output = model(
            token_encoder_inputs=batch["token_encoder_inputs"],
            type_encoder_inputs=batch.get("type_encoder_inputs"),
            labels=batch["labels"],
        )
    assert output.span_logits is not None, f"{architecture}: no span logits"
    print(f"  {architecture}: span logits {tuple(output.span_logits.shape)}")


def main():
    silence_transformers_warnings()
    torch.manual_seed(0)

    print("configs:")
    check_configs()

    print("architectures:")
    with tempfile.TemporaryDirectory() as tmp:
        encoder = make_tiny_encoder(Path(tmp))
        for architecture in ARCHITECTURE_REGISTRY:
            check_architecture(architecture, encoder)

    print("\nsmoke test passed")


if __name__ == "__main__":
    sys.exit(main())
