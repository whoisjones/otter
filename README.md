# 🦦 Otter - Universal Multilingual NER

## Reproducing Paper Results

## Input Format

Datasets need to be annotated with character offsets, following this naming convention:

```python
dataset = DatasetDict({
    "test": Dataset.from_list([
        {
            "text": "John Doe works at OpenAI in San Francisco.",
            "char_spans": [
                {"start": 0, "end": 8, "label": "person"},
                {"start": 18, "end": 24, "label": "organization"},
                {"start": 28, "end": 41, "label": "location"},
            ]
        },
        {
            "text": "Alice and Bob visited the Eiffel Tower.",
            "char_spans": [
                {"start": 0, "end": 5, "label": "person"},
                {"start": 10, "end": 13, "label": "person"},
                {"start": 28, "end": 40, "label": "location"},
            ]
        },
        {
            "text": "Amazon was founded by Jeff Bezos.",
            "char_spans": [
                {"start": 0, "end": 6, "label": "organization"},
                {"start": 22, "end": 32, "label": "person"},
            ]
        }
    ])
})
```

You can also use word-segmented inputs and labels using the names `tokens` and `token_spans`.

## Usage

The four released models are self-contained: `predict` takes raw strings and a list of
entity types in plain language, and returns character spans.

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

Pass a list of strings to run on a batch; you then get one list of entities per input,
in the same order.

```python
model = model.to("cuda")
results = model.predict(texts, labels=["person", "location"], batch_size=16, threshold=0.5)
```

`threshold` defaults to `config.prediction_threshold` -- 0.5 for the cross-encoders and
0.2 for the bi-encoders, from the macro-F1 calibration sweep.

| Model | Architecture | Encoder |
|---|---|---|
| [`whoisjones/otter-bi-mmbert`](https://huggingface.co/whoisjones/otter-bi-mmbert) | bi-encoder | mmBERT-base |
| [`whoisjones/otter-cross-mmbert`](https://huggingface.co/whoisjones/otter-cross-mmbert) | cross-encoder | mmBERT-base |
| [`whoisjones/otter-bi-rembert`](https://huggingface.co/whoisjones/otter-bi-rembert) | bi-encoder | RemBERT |
| [`whoisjones/otter-cross-rembert`](https://huggingface.co/whoisjones/otter-cross-rembert) | cross-encoder | RemBERT |

### Publishing the models

`scripts/hf_checkpoint_conversion/prepare_hub_repos.py` rebuilds the Hub repositories
from `scripts/hf_checkpoint_conversion/hub_files/`. It stages the config, the
`trust_remote_code` modules, the tokenizers and the model card, and leaves the published
weights untouched.

```bash
python scripts/hf_checkpoint_conversion/prepare_hub_repos.py --out-dir build/hub
python scripts/hf_checkpoint_conversion/prepare_hub_repos.py --out-dir build/hub --push
```

## Training

All training is launched through the single entry point `train.py`. The
architecture is selected via the `architecture` field of the config JSON
(`bi_encoder`, `cross_encoder`, `contrastive_bi_encoder`, or
`contrastive_cross_encoder`).

### Bi-Encoder Models

**BCE Loss:**
```bash
accelerate launch train.py configs/bi_encoder.json
```

**Contrastive Loss:**
```bash
accelerate launch train.py configs/bi_encoder_contrastive.json
```

### Cross-Encoder Models

**BCE Loss:**
```bash
accelerate launch train.py configs/cross_encoder.json
```

**Contrastive Loss:**
```bash
accelerate launch train.py configs/cross_encoder_contrastive.json
```

### Customizing Training Data

To use multiple training files (e.g., all finerweb files), first download the dataset from the hub:
```python
from datasets import load_dataset
dataset = load_dataset('whoisjones/finerweb', "eng", split='train')
dataset.to_json('data/finerweb/train.jsonl')
```

Then modify the config:
```json
{
  "train_file": "data/finerweb/*.jsonl",
  "validation_file": "data/conll2003/validation.jsonl",
  "test_file": "data/conll2003/test.jsonl"
}
```

The `train_file` field supports glob patterns, so `*.jsonl` will match all JSONL files in the directory.

To change the test dataset, simply update the `test_file` path in the config to point to your desired evaluation dataset.

## Evaluation

All evaluation is launched through the single entry point `evaluate.py`. The
architecture is inferred from the checkpoint's config.

```bash
python evaluate.py \
  --pretrained_model_name_or_path models/bi_encoder/best_checkpoint \
  --evaluation_dataset data/conll2003/test.jsonl \
  --threshold 0.5 \
  --evaluation_format tokens
```

### Evaluation Dataset Formats

The `--evaluation_dataset` argument accepts:
- **JSONL files**: Path to a `.jsonl` file (e.g., `data/conll2003/test.jsonl`)
- **HuggingFace DatasetDict**: Path to a directory containing a saved DatasetDict (e.g., `data/eval_data/panx/en`)

The script automatically detects the format and loads the appropriate split (`test` or `dev`).

### Evaluation Format

- `--evaluation_format text`: Uses character-level spans (`char_spans`) from the dataset
- `--evaluation_format tokens`: Uses token-level spans (`token_spans`) from the dataset

### Threshold

- For **BCE models**: Pass a float value (e.g., `0.5`)
- For **Contrastive models**: Pass either `"cls"` or `"label_token"` as a string
