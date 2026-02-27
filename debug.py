import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from collate_fn import AllLabelsCollator, InBatchNegativesCollator # import this file from the model repository
from src.config import SpanModelConfig
from src.model import OtterBiEncoderModel
from datasets import DatasetDict, Dataset, load_dataset

def eval():
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

def train():
    dataset = load_dataset('whoisjones/finerweb', "eng", split="train")

    config = SpanModelConfig(token_encoder="google-bert/bert-base-uncased", type_encoder="google-bert/bert-base-uncased")
    model = OtterBiEncoderModel(config=config).to("cuda")
    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_collator = InBatchNegativesCollator(token_encoder_tokenizer, type_encoder_tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=train_collator)
    
    max_steps = 500
    step = 0
    while step < max_steps:
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            optimizer.zero_grad()

            token_encoder_inputs = {k: v.to("cuda") for k, v in batch["token_encoder_inputs"].items()}
            type_encoder_inputs = {k: v.to("cuda") for k, v in batch["type_encoder_inputs"].items()}
            labels = {k: v.to("cuda") for k, v in batch["labels"].items()}
            outputs = model(
                token_encoder_inputs=token_encoder_inputs,
                type_encoder_inputs=type_encoder_inputs,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            avg_loss = total_loss / num_batches
            if step % 10 == 0:
                print(f"Step {step }: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")
            step += 1
        print(f"Training complete! Average Loss: {avg_loss:.4f}")

def publish_datasets():
    import glob
    name = "multinerd"
    # print('configs:')
    # for path in glob.glob(f"/vol/tmp/goldejon/ner/eval_data/{name}/*"):
    #     dataset = DatasetDict.load_from_disk(path)
    #     language = path.split("/")[-1]
    #     print(f"- config_name: {language}")
    #     print(f"  data_files:")
    #     if "train" in dataset:
    #         dataset["train"].to_json(f"data/{name}/{language}_train.jsonl")
    #         print(f"  - split: train")
    #         print(f"    path: data/{language}_train*")
    #     if "dev" in dataset:
    #         dataset["dev"].to_json(f"data/{name}/{language}_dev.jsonl")
    #         print(f"  - split: dev")
    #         print(f"    path: data/{language}_dev*")
    #     if "test" in dataset:
    #         dataset["test"].to_json(f"data/{name}/{language}_test.jsonl")
    #         print(f"  - split: test")
    #         print(f"    path: data/{language}_test*")

    # data_files = {p.split("/")[-1].split(".")[0]: p for p in glob.glob(f"data/{name}/*")}
    # dataset = load_dataset("json", data_files=data_files)
    # dataset.push_to_hub(f"whoisjones/{name}")

    dataset = load_dataset(f"whoisjones/masakhaner", "swa", split="train")
    print()

if __name__ == "__main__":
    from src.model import OtterBiEncoderModel
    from transformers import AutoModelForTokenClassification
    model = OtterBiEncoderModel.from_pretrained("whoisjones/otter-bi-mmbert")
    model = AutoModelForTokenClassification.from_pretrained("whoisjones/otter-bi-mmbert", trust_remote_code=True)
    print(model)