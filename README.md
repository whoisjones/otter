# What matters when building universal multiingual NER models?

This is the repository to our EMNLP paper in which we empirically compare various design choices to train universal NER models.
Otter is the result of this: a span-based named entity recognizer that takes the entity types as *input*.
Instead of a fixed tag set, you pass the types you care about in plain language —
`person`, `Fußballverein`, `protein` — and the model returns the matching character
spans.

Four pretrained models are published on the Hugging Face Hub, and this repository
contains everything needed to train, evaluate and publish them.

| Model | Architecture | Encoder | Max length |
|---|---|---|---|
| [`whoisjones/otter-bi-mmbert`](https://huggingface.co/whoisjones/otter-bi-mmbert) | bi-encoder | mmBERT-base | 1024 |
| [`whoisjones/otter-cross-mmbert`](https://huggingface.co/whoisjones/otter-cross-mmbert) | cross-encoder | mmBERT-base | 1024 |
| [`whoisjones/otter-bi-rembert`](https://huggingface.co/whoisjones/otter-bi-rembert) | bi-encoder | RemBERT | 512 |
| [`whoisjones/otter-cross-rembert`](https://huggingface.co/whoisjones/otter-cross-rembert) | cross-encoder | RemBERT | 512 |

The **bi-encoder** embeds the text and the type names with two separate encoders and
matches spans against types in a shared space — the type embeddings can be cached, so
it stays fast with many types. The **cross-encoder** puts the type names and the text
through one encoder as `[LABEL] person [LABEL] location [SEP] <text>`, which is more
accurate but re-encodes the prefix for every input.

## Installation

The project is defined entirely by `pyproject.toml`.

```bash
uv sync                     # or: uv pip install -e .
uv pip install -e ".[dev]"  # adds ruff
```

## Quickstart

The released models are self-contained: `predict` takes raw strings plus a list of
entity types and returns character spans.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("whoisjones/otter-cross-mmbert", trust_remote_code=True)
model.eval()

entities = model.predict(
    "Angela Merkel besuchte gestern das Brandenburger Tor in Berlin.",
    labels=["person", "organization", "location"],
)

for entity in entities:
    print(f"{entity['text']!r:25} {entity['label']:15} {entity['score']:.2f}")
```

```
'Angela Merkel'           person          0.99
'Brandenburger Tor'       location        0.82
'Berlin'                  location        0.90
```

The same call works in any of the supported languages -- the labels are free text and
do not have to be in the language of the input:

```python
entities = model.predict(
    "马云在杭州创办了阿里巴巴集团。",
    labels=["person", "organization", "location"],
)
```

```
'马云'                    person          0.95
'杭州'                    location        0.84
'阿里巴巴集团'              organization    0.85
```

Scripts written without spaces between words -- Chinese, Japanese, Thai -- need no
segmentation step: pass the sentence as it is written. Do not insert spaces between
characters, and if your input arrives pre-tokenised, join it back into natural text
first. A model given `马 云 在 杭 州` reads it as unrelated characters and scores far
worse than on `马云在杭州`.

Pass a list of strings to run on a batch; you get one list of entities per input, in
the same order.

```python
model = model.to("cuda")
results = model.predict(texts, labels=["person", "location"], batch_size=16, threshold=0.5)
```

`threshold` defaults to `config.prediction_threshold` — 0.5 for the cross-encoders and
0.2 for the bi-encoders, taken from the macro-F1 calibration sweep.

## Repository layout

```
train.py            training entry point, one config JSON per run
evaluate.py         evaluation entry point, architecture inferred from the checkpoint
publish_to_hub.py   builds and pushes the four Hub repositories
configs/            one JSON per architecture
src/args.py         ModelArguments / DataTrainingArguments / CustomTrainingArguments
src/config.py       SpanModelConfig, shared by all four architectures
src/model/          bi-encoder, cross-encoder and their contrastive variants
src/collator/       span enumeration, masking and label alignment
src/loss.py         BCE, focal, dice, dice+focal and contrastive losses
src/metrics.py      span decoding and micro/macro P/R/F1
src/sampling.py     language-weighted sampling of the training mixture
src/trainer.py      training and evaluation loops
src/hub/            sources for the published `trust_remote_code` model repositories
```

## Data format

Datasets are JSONL, annotated with character offsets:

```python
{
    "text": "John Doe works at OpenAI in San Francisco.",
    "char_spans": [
        {"start": 0, "end": 8, "label": "person"},
        {"start": 18, "end": 24, "label": "organization"},
        {"start": 28, "end": 41, "label": "location"},
    ],
}
```

Word-segmented input works too, under the keys `tokens` and `token_spans`, where the
offsets index words rather than characters. Which of the two is used is decided by
`annotation_format` (`text` or `tokens`) in the config.

## Training

All training goes through `train.py`, with the architecture selected by the
`architecture` field of the config JSON.

```bash
accelerate launch train.py configs/bi_encoder.json
accelerate launch train.py configs/bi_encoder_contrastive.json
accelerate launch train.py configs/cross_encoder.json
accelerate launch train.py configs/cross_encoder_contrastive.json
```

The BCE configs use `loss_fn: "bce"`; the contrastive configs pair the
`contrastive_*` architectures with `loss_fn: "contrastive"`.

Training data is pointed at by `train_file`, which accepts glob patterns:

```python
from datasets import load_dataset

dataset = load_dataset("whoisjones/finerweb", "eng", split="train")
dataset.to_json("data/finerweb/eng.jsonl")
```

```json
{
  "train_file": "data/finerweb/*.jsonl",
  "validation_file": "data/conll2003/validation.jsonl",
  "test_file": "data/conll2003/test.jsonl"
}
```

Checkpoints land in `output_dir`; the best one by validation F1 is symlinked as
`best_checkpoint`. A run whose `best_checkpoint` already exists exits immediately, so
re-launching a finished job is a no-op.

### Controlling the language mixture

By default every sentence is equally likely to be sampled, so each language contributes
in proportion to how much of it there is. Across the 91 languages of FiNERweb that
spans 7k to 58k sentences, and the languages with the least data are often the ones that
need the most exposure. `language_weights` multiplies a language's share of the draws:

```json
{
  "train_file": "data/finerweb/*.jsonl",
  "language_weights": {"tha": 10, "khm": 10, "cmn": 3}
}
```

The language is read from the prefix of each example id (`tha_2200-s0`), which is how
the per-language FiNERweb files are keyed. Languages left out keep weight 1, and
omitting the field entirely samples uniformly.

A weight is a multiplier on existing mass, not a target share, so what a language ends
up with depends on how much data it already has. Check before committing to a run:

```python
from datasets import load_dataset
from src.sampling import language_mixture

dataset = load_dataset("json", data_files="data/finerweb/*.jsonl", split="train")
mixture = language_mixture(dataset, {"tha": 10, "khm": 10, "cmn": 3})
print({lang: f"{share:.1%}" for lang, share in mixture.items() if share > 0.02})
```

Weight 10 lifts Thai from 1.7% of FiNERweb to about 12%. `train.py` logs the resulting
shares at startup, so every run records its own mixture. Naming a language that is not
in the training data is an error rather than a silent no-op.

Sampling is with replacement: an upweighted language repeats within an epoch rather
than the other languages being dropped. Upweighting a few languages heavily does take
exposure away from the rest, so evaluate on languages you did not touch as well.

## Evaluation

`evaluate.py` infers the architecture from the checkpoint config.

```bash
python evaluate.py \
  --pretrained_model_name_or_path models/bi_encoder/best_checkpoint \
  --evaluation_dataset data/conll2003/test.jsonl \
  --threshold 0.5 \
  --evaluation_format tokens
```

- `--evaluation_dataset` takes a `.jsonl` file, a directory holding a saved
  `DatasetDict`, or a Hub dataset name (every config is then evaluated in turn). The
  `test` split is used, falling back to `dev`.
- `--evaluation_format` picks `char_spans` (`text`) or `token_spans` (`tokens`).
- `--threshold` is a float for the BCE models, or `cls` / `label_token` for the
  contrastive ones. Omit it to use the checkpoint's own `prediction_threshold`.
- `--max_eval_samples` caps the split size (default 1000, sampled with seed 42).

Results are written to `evals/eval_<timestamp>.json` alongside VRAM, latency and FLOP
measurements.

## Publishing to the Hub

`publish_to_hub.py` rebuilds the four model repositories. It stages the config, the
`trust_remote_code` modules from `src/hub/`, the tokenizers and the model card, and
generates `masks.py`, `loss.py`, `metrics.py` and `collate_fn.py` from `src/` so the
published copies cannot drift from this repository. Published weights are never
touched.

```bash
python publish_to_hub.py --out-dir build/hub            # stage only
python publish_to_hub.py --out-dir build/hub --example  # also run the model-card example
python publish_to_hub.py --out-dir build/hub --example --push
```

## Development

```bash
ruff format .        # formatting
ruff check .         # linting
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
