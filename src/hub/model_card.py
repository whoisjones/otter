ARCH_BLURB = {
    "bi_encoder": (
        "**Bi-encoder.** The text and the entity type names are encoded separately, and "
        "candidate spans are scored against the label embeddings. Label embeddings depend "
        "only on the label set, so they can be computed once and reused across a whole "
        "corpus -- the cheaper option when the same types are applied to many inputs."
    ),
    "cross_encoder": (
        "**Cross-encoder.** The entity types are prepended to the text as a "
        "`[LABEL] <type> ... [SEP] ` prefix, so a single encoder sees labels and text "
        "together and they attend to each other. More accurate than the bi-encoder, at the "
        "cost of re-encoding the text for every label set."
    ),
}

EXTRA = {
    "bi_encoder": """
### Reusing label embeddings

The type encoder only sees the label names, so its output can be computed once and
reused for every batch:

```python
type_inputs = model.encode_labels(labels)   # do this once
```

`predict` does this per call; drop down to `forward` if you are running over a large
corpus with a fixed label set.
""",
    "cross_encoder": """
### Prompt format

The model is trained on inputs of the form

```
[LABEL] person [LABEL] organization [SEP] John Doe works at OpenAI.
```

`model.build_prompt(labels)` returns that prefix if you want to build inputs yourself.
Note that the prefix counts against `max_seq_length`, so very long label sets leave
less room for the text.
""",
}


EXAMPLE_TEXT = "Angela Merkel besuchte gestern das Brandenburger Tor in Berlin."
EXAMPLE_LABELS = ["person", "organization", "location"]


def render(name, spec, threshold, example_output=None):
    architecture = spec["architecture"]
    encoders = (
        f"- Text encoder: [`{spec['token_encoder']}`](https://huggingface.co/{spec['token_encoder']})\n"
        f"- Label encoder: [`{spec['type_encoder']}`](https://huggingface.co/{spec['type_encoder']})\n"
        if architecture == "bi_encoder"
        else f"- Encoder: [`{spec['token_encoder']}`](https://huggingface.co/{spec['token_encoder']}) "
        "(vocabulary extended with a `[LABEL]` token)\n"
    )

    if example_output:
        example_block = "\n```\n" + example_output.rstrip() + "\n```\n"
    else:
        example_block = ""

    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
tags:
- named-entity-recognition
- ner
- zero-shot
- multilingual
- otter
language:
- multilingual
---

# 🦦 {name}

Otter is a multilingual, open-type named entity recognizer. You give it a piece of text
and a list of entity types in plain language -- `["person", "band", "chemical compound"]`
-- and it returns the character spans of the entities of those types. There is no fixed
label set and no fine-tuning step: the types are part of the input.

{ARCH_BLURB[architecture]}

{encoders}- Max sequence length: {spec["max_seq_length"]} tokens
- Max span length: 30 tokens

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("whoisjones/{name}", trust_remote_code=True)
model.eval()

entities = model.predict(
    "{EXAMPLE_TEXT}",
    labels={EXAMPLE_LABELS},
)

for entity in entities:
    print(f"{{entity['text']!r:25}} {{entity['label']:15}} {{entity['score']:.2f}}")
```
{example_block}
Each entity is a dict with `text`, `label`, `start`, `end` (character offsets into the
input string) and `score`. Pass a list of strings to run on a batch; you then get one
list of entities per input, in the same order:

```python
model = model.to("cuda")

texts = ["Angela Merkel besuchte das Brandenburger Tor.", "Sony was founded in Tokyo."]
results = model.predict(texts, labels=["person", "organization", "location"], batch_size=16)
```

### Threshold

`predict` keeps spans scoring above `threshold`, which defaults to
`config.prediction_threshold` (**{threshold}** for this checkpoint, chosen by calibrating
macro-F1 across the evaluation suite). Lower it for higher recall, raise it for higher
precision:

```python
entities = model.predict(text, labels=labels, threshold=0.1)
```

Because the label set is part of the input, the useful threshold shifts with how many
types you ask for and how specific they are. If you have a few hundred annotated
sentences from your own domain, re-calibrating on those is worth more than any default.

### Writing good label names

The label is read as natural language, so it carries meaning. `"politician"` and
`"person"` select different spans, and a phrase like `"chemical compound"` works as well
as a single word. Prefer the wording you would use to describe the type to a person.
{EXTRA[architecture]}
## Fine-tuning

`collate_fn.py` in this repository holds the training and evaluation collators. See the
[GitHub repository](https://github.com/whoisjones/otter) for the full training pipeline,
the evaluation suite, and the data preparation scripts.

## Model family

| Model | Architecture | Encoder |
|---|---|---|
| [`whoisjones/otter-bi-mmbert`](https://huggingface.co/whoisjones/otter-bi-mmbert) | bi-encoder | mmBERT-base |
| [`whoisjones/otter-cross-mmbert`](https://huggingface.co/whoisjones/otter-cross-mmbert) | cross-encoder | mmBERT-base |
| [`whoisjones/otter-bi-rembert`](https://huggingface.co/whoisjones/otter-bi-rembert) | bi-encoder | RemBERT |
| [`whoisjones/otter-cross-rembert`](https://huggingface.co/whoisjones/otter-cross-rembert) | cross-encoder | RemBERT |

The cross-encoders are the stronger models; the bi-encoders are cheaper when one label
set is applied across a large corpus.

## License

Apache 2.0.
"""
