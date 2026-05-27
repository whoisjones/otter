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

You can use our off-the-shelf models from the HF hub but you need to download the [collator file](https://huggingface.co/whoisjones/otter-bi-mmbert/blob/main/collate_fn.py) and put it into the project directory.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from collate_fn import AllLabelsCollator # import this file from the model repository
from datasets import DatasetDict, Dataset

def main():
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

    config = AutoConfig.from_pretrained("whoisjones/otter-bi-mmbert", trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained("whoisjones/otter-bi-mmbert", trust_remote_code=True)
    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)

    labels = list(set([span["label"] for sample in dataset["test"] for span in sample["char_spans"]]))
    label2id = {label: idx for idx, label in enumerate(labels)}
    collator = AllLabelsCollator(token_encoder_tokenizer, type_encoder_tokenizer, label2id=label2id)
    dataloader = DataLoader(dataset["test"], batch_size=1, collate_fn=collator)

    for batch in dataloader:
        gold_labels = batch["labels"]["ner"]
        predictions = model.predict(batch, threshold=0.1)
        print(f"Gold labels: {gold_labels}")
        print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
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
