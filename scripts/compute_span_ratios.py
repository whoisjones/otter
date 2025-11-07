from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

def all_spans_mask(input_ids, sequence_ids):
    text_start_index = 0
    while sequence_ids[text_start_index] == None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] == None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    for sequence_id in sequence_ids:
        start_mask.append(1 if sequence_id is not None else 0)
        end_mask.append(1 if sequence_id is not None else 0)

    span_mask = [
        [
            (j - i >= 0) * s * e for j, e in enumerate(end_mask)
        ]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask

def subwords_mask(input_ids, word_ids):
    text_start_index = 0
    while word_ids[text_start_index] == None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while word_ids[text_end_index] == None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            start_mask.append(1 if word_id != previous_word_id else 0)
            end_mask.append(1 if (idx + 1 == len(word_ids) or word_ids[idx + 1] != word_id) else 0)
        else:
            start_mask.append(0)
            end_mask.append(0)
        previous_word_id = word_id

    span_mask = [
        [
            (j - i >= 0) * s * e for j, e in enumerate(end_mask)
        ]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask

def average(scores):
    return sum(scores) / len(scores)



def main():

    def collate_fn(batch):
        if format == 'text':
            texts = [sample['text'] for sample in batch]
        elif format == 'tokens':
            texts = [sample["tokens"] for sample in batch]
        else:
            raise ValueError(f"Invalid format: {format}")

        avg_length = sum(len(text) for text in texts) / len(texts)
        
        token_encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True if format == 'text' else False,
            is_split_into_words=True if format == 'tokens' else False
        )

        unique_types = []
        for sample in batch:
            for span in sample["token_spans" if format == 'tokens' else "char_spans"]:
                if span["label"] not in unique_types:
                    unique_types.append(span["label"])
        unique_types = sorted(unique_types)
        type2id_batch = {entity_type: idx for idx, entity_type in enumerate(unique_types)}

        if format == 'text':
            offset_mapping = token_encodings.pop("offset_mapping")

        annotations = {
            "avg_length": avg_length,
            "start_labels": [],
            "end_labels": [],
            "span_labels": [],
            "start_loss_mask": [],
            "end_loss_mask": [],
            "span_loss_mask": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]

            if loss_masking == 'subwords':
                word_ids = token_encodings.word_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask = subwords_mask(input_ids, word_ids)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask = all_spans_mask(input_ids, sequence_ids)

            start_loss_mask = torch.tensor([start_mask[:] for _ in unique_types])
            end_loss_mask = torch.tensor([end_mask[:] for _ in unique_types])
            span_loss_mask = torch.tensor([[x[:] for x in span_mask] for _ in unique_types])

            start_labels = torch.zeros(len(unique_types), len(input_ids))
            end_labels = torch.zeros(len(unique_types), len(input_ids))
            span_labels = torch.zeros(len(unique_types), len(input_ids), len(input_ids))

            for label in sample_labels:
                if format == 'text':
                    offsets = offset_mapping[i]
                    if offsets[text_start_index][0] <= label["start"] and offsets[text_end_index][1] >= label["end"]:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"]:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"]:
                            end_label_index -= 1
                        end_label_index += 1

                        start_labels[type2id_batch[label["label"]], start_label_index] = 1
                        end_labels[type2id_batch[label["label"]], end_label_index] = 1
                        span_labels[type2id_batch[label["label"]], start_label_index, end_label_index] = 1

                elif format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] in word_ids and label["end"] - 1 in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"]:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"]:
                            end_label_index -= 1

                        start_labels[type2id_batch[label["label"]], start_label_index] = 1
                        end_labels[type2id_batch[label["label"]], end_label_index] = 1
                        span_labels[type2id_batch[label["label"]], start_label_index, end_label_index] = 1

            annotations["start_labels"].append(start_labels)
            annotations["end_labels"].append(end_labels)
            annotations["span_labels"].append(span_labels)
            annotations["start_loss_mask"].append(start_loss_mask)
            annotations["end_loss_mask"].append(end_loss_mask)
            annotations["span_loss_mask"].append(span_loss_mask)

        annotations["start_labels"] = torch.stack(annotations["start_labels"], dim=0).sum().numpy().item()
        annotations["end_labels"] = torch.stack(annotations["end_labels"], dim=0).sum().numpy().item()
        annotations["start_loss_mask"] = torch.stack(annotations["start_loss_mask"], dim=0).sum().numpy().item()
        annotations["end_loss_mask"] = torch.stack(annotations["end_loss_mask"], dim=0).sum().numpy().item()
        annotations["span_labels"] = torch.stack(annotations["span_labels"], dim=0).sum().numpy().item()
        annotations["span_loss_mask"] = torch.stack(annotations["span_loss_mask"], dim=0).sum().numpy().item()

        return annotations

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    for format in ['text', 'tokens']:
        for loss_masking in ['none', 'subwords']:
            for dataset_name in ['pilener', 'nuner']:
                # Limit to max 2500 samples
                dataset = load_dataset(
                    "json",
                    data_files={
                        "train": f"/vol/tmp/goldejon/ner/data/{dataset_name}/eng.jsonl",
                    }
                )
                if len(dataset["train"]) > 2500:
                    dataset["train"] = dataset["train"].select(range(2500))

                results = {}

                for split in dataset:
                    dataloader = DataLoader(dataset[split], batch_size=2, collate_fn=collate_fn)
                    scores = {
                        'avg_length': [],
                        'start_labels': [],
                        'end_labels': [],
                        'span_labels': [],
                        'start_loss_mask': [],
                        'end_loss_mask': [],
                        'span_loss_mask': []
                    }
                    for batch in tqdm(dataloader):
                        scores['avg_length'].append(batch['avg_length'])
                        scores['start_labels'].append(batch['start_labels'])
                        scores['end_labels'].append(batch['end_labels'])
                        scores['span_labels'].append(batch['span_labels'])
                        scores['start_loss_mask'].append(batch['start_loss_mask'])
                        scores['end_loss_mask'].append(batch['end_loss_mask'])
                        scores['span_loss_mask'].append(batch['span_loss_mask'])
                    means = {k: average(v) for k, v in scores.items()}
                    means['format'] = format
                    means['loss_masking'] = loss_masking
                    means['dataset'] = f'finerweb_{dataset_name}'
                    results[split] = means

                    with open('span_ratios.jsonl', 'a') as f:
                        f.write(json.dumps(means) + '\n')

if __name__ == "__main__":
    main()

