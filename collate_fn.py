import random
import torch

def subword_span_mask(input_ids, sequence_ids, max_span_length):
    num_tokens = len(input_ids)
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

    spans_idx = [
        (i, i + j)
        for i in range(num_tokens)
        for j in range(max_span_length)
        if i + j < num_tokens
    ]
    span_lengths = [end_subword_index - start_subword_index + 1 for start_subword_index, end_subword_index in spans_idx]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index >= text_start_index
            and end_subword_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths

def word_segmented_span_mask(input_ids, word_ids, max_span_length):
    num_tokens = len(input_ids)
    text_start_index = 0
    while text_start_index < num_tokens and word_ids[text_start_index] is None:
        text_start_index += 1

    text_end_index = num_tokens - 1
    while text_end_index >= 0 and word_ids[text_end_index] is None:
        text_end_index -= 1
        
    start_mask = []
    end_mask = []
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            is_start = 1 if word_id != previous_word_id else 0

            if idx + 1 == num_tokens or word_ids[idx + 1] != word_id:
                is_end = 1
            else:
                is_end = 0
        else:
            is_start = 0
            is_end = 0

        start_mask.append(is_start)
        end_mask.append(is_end)
        previous_word_id = word_id

    spans_idx = [
        (i, i + j)
        for i in range(num_tokens)
        for j in range(max_span_length)
        if i + j < num_tokens
    ]
    span_lengths = [end_subword_index - start_subword_index + 1 for start_subword_index, end_subword_index in spans_idx]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index >= text_start_index
            and end_subword_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths

class InBatchNegativesCollator:
    def __init__(
        self, 
        token_encoder_tokenizer, 
        type_encoder_tokenizer, 
        max_seq_length=512, 
        max_span_length=30, 
        format='text', 
    ):
        self.token_encoder_tokenizer = token_encoder_tokenizer
        self.type_encoder_tokenizer = type_encoder_tokenizer
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.format = format

    def __call__(self, batch):
        if self.format == 'text':
            texts = [sample['text'] for sample in batch]
        elif self.format == 'tokens':
            texts = [sample["tokens"] for sample in batch]
        else:
            raise ValueError(f"Invalid format: {self.format}")
        
        token_encodings = self.token_encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True if self.format == 'text' else False,
            is_split_into_words=True if self.format == 'tokens' else False
        )
        
        unique_types = []
        for sample in batch:
            for span in sample["token_spans" if self.format == 'tokens' else "char_spans"]:
                if span["label"] not in unique_types:
                    unique_types.append(span["label"])
        random.shuffle(unique_types)
        type2id_batch = {entity_type: idx for idx, entity_type in enumerate(unique_types)}
        
        if not unique_types:
            return {}
        
        type_encodings = self.type_encoder_tokenizer(
            unique_types,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        if self.format == 'text':
            offset_mapping = token_encodings.pop("offset_mapping")

        annotations = {
            "start_labels": [],
            "end_labels": [],
            "span_labels": [],
            "valid_start_mask": [],
            "valid_end_mask": [],
            "valid_span_mask": [],
            "span_subword_indices": [],
            "span_lengths": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]

            if self.format == 'tokens':
                word_ids = token_encodings.word_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = word_segmented_span_mask(input_ids, word_ids, self.max_span_length)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = subword_span_mask(input_ids, sequence_ids, self.max_span_length)

            span_lookup = {span: idx for idx, span in enumerate(spans_idx)}

            valid_start_mask = torch.tensor([start_mask[:] for _ in unique_types])
            valid_end_mask = torch.tensor([end_mask[:] for _ in unique_types])
            valid_span_mask = torch.tensor([span_mask[:] for _ in unique_types])
            span_subword_indices = torch.tensor(spans_idx)
            span_lengths = torch.tensor(span_lengths)

            start_labels = torch.zeros(len(unique_types), len(input_ids))
            end_labels = torch.zeros(len(unique_types), len(input_ids))
            span_labels = torch.zeros(len(unique_types), len(spans_idx))

            for label in sample_labels:
                if self.format == 'text':
                    offsets = offset_mapping[i]
                    if offsets[text_start_index][0] <= label["start"] and offsets[text_end_index][1] >= label["end"]:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"]:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"]:
                            end_label_index -= 1
                        end_label_index += 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[type2id_batch[label["label"]], start_label_index] = 1
                        end_labels[type2id_batch[label["label"]], end_label_index] = 1
                        span_labels[type2id_batch[label["label"]], span_lookup[(start_label_index, end_label_index)]] = 1

                elif self.format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] in word_ids and label["end"] - 1 in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"]:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"]:
                            end_label_index -= 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[type2id_batch[label["label"]], start_label_index] = 1
                        end_labels[type2id_batch[label["label"]], end_label_index] = 1
                        span_labels[type2id_batch[label["label"]], span_lookup[(start_label_index, end_label_index)]] = 1

            annotations["start_labels"].append(start_labels)
            annotations["end_labels"].append(end_labels)
            annotations["span_labels"].append(span_labels)
            annotations["valid_start_mask"].append(valid_start_mask)
            annotations["valid_end_mask"].append(valid_end_mask)
            annotations["valid_span_mask"].append(valid_span_mask)
            annotations["span_subword_indices"].append(span_subword_indices)
            annotations["span_lengths"].append(span_lengths)

        annotations["start_labels"] = torch.stack(annotations["start_labels"], dim=0)
        annotations["end_labels"] = torch.stack(annotations["end_labels"], dim=0)
        annotations["span_labels"] = torch.stack(annotations["span_labels"], dim=0)
        annotations["valid_start_mask"] = torch.stack(annotations["valid_start_mask"], dim=0)
        annotations["valid_end_mask"] = torch.stack(annotations["valid_end_mask"], dim=0)
        annotations["valid_span_mask"] = torch.stack(annotations["valid_span_mask"], dim=0)
        annotations["span_subword_indices"] = torch.stack(annotations["span_subword_indices"], dim=0)
        annotations["span_lengths"] = torch.stack(annotations["span_lengths"], dim=0)

        token_encoder_inputs = {
            "input_ids": token_encodings["input_ids"],
            "attention_mask": token_encodings["attention_mask"]
        }
        if "token_type_ids" in token_encodings:
            token_encoder_inputs["token_type_ids"] = token_encodings["token_type_ids"]
        
        type_encoder_inputs = {
            "input_ids": type_encodings["input_ids"],
            "attention_mask": type_encodings["attention_mask"]
        }
        if "token_type_ids" in type_encodings:
            type_encoder_inputs["token_type_ids"] = type_encodings["token_type_ids"]

        batch = {
            "token_encoder_inputs": token_encoder_inputs,
            "type_encoder_inputs": type_encoder_inputs,
            "labels": annotations
        }

        return batch

class AllLabelsCollator:
    def __init__(
        self, 
        token_encoder_tokenizer,
        type_encoder_tokenizer,
        label2id, 
        max_seq_length=512, 
        max_span_length=30, 
        format='text', 
    ):
        self.token_encoder_tokenizer = token_encoder_tokenizer
        type_encodings = type_encoder_tokenizer(
            list(label2id.keys()),
            truncation=True,
            max_length=64,
            padding="longest" if len(label2id) <= 1000 else "max_length",
            return_tensors="pt"
        )
        self.type_input_ids = type_encodings["input_ids"]
        self.type_attention_mask = type_encodings["attention_mask"]
        self.type_token_type_ids = type_encodings["token_type_ids"] if "token_type_ids" in type_encodings else None
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.format = format

    def __call__(self, batch):
        if self.format == 'text':
            texts = [sample['text'] for sample in batch]
        elif self.format == 'tokens':
            texts = [sample["tokens"] for sample in batch]
        else:
            raise ValueError(f"Invalid format: {self.format}")
        
        token_encodings = self.token_encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True if self.format == 'text' else False,
            is_split_into_words=True if self.format == 'tokens' else False
        )
        
        if self.format == 'text':
            offset_mapping = token_encodings.pop("offset_mapping")

        annotations = {
            "ner": [],
            "start_labels": [],
            "end_labels": [],
            "span_labels": [],
            "valid_start_mask": [],
            "valid_end_mask": [],
            "valid_span_mask": [],
            "span_subword_indices": [],
            "span_lengths": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]

            if self.format == 'tokens':
                word_ids = token_encodings.word_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = word_segmented_span_mask(input_ids, word_ids, self.max_span_length)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = subword_span_mask(input_ids, sequence_ids, self.max_span_length)

            span_lookup = {span: idx for idx, span in enumerate(spans_idx)}

            valid_start_mask = torch.tensor([start_mask[:] for _ in range(len(self.label2id))])
            valid_end_mask = torch.tensor([end_mask[:] for _ in range(len(self.label2id))])
            valid_span_mask = torch.tensor([span_mask[:] for _ in range(len(self.label2id))])
            span_subword_indices = torch.tensor(spans_idx)
            span_lengths = torch.tensor(span_lengths)

            start_labels = torch.zeros(len(self.label2id), len(input_ids))
            end_labels = torch.zeros(len(self.label2id), len(input_ids))
            span_labels = torch.zeros(len(self.label2id), len(spans_idx))

            annotation = []

            for label in sample_labels:
                if self.format == 'text':
                    offsets = offset_mapping[i]
                    if offsets[text_start_index][0] <= label["start"] and offsets[text_end_index][1] >= label["end"]:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"]:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"]:
                            end_label_index -= 1
                        end_label_index += 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[self.label2id[label["label"]], start_label_index] = 1
                        end_labels[self.label2id[label["label"]], end_label_index] = 1
                        span_labels[self.label2id[label["label"]], span_lookup[(start_label_index, end_label_index)]] = 1

                        annotation.append({
                            "start": start_label_index,
                            "end": end_label_index,
                            "label": label["label"]
                        })

                elif self.format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] in word_ids and label["end"] - 1 in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"]:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"]:
                            end_label_index -= 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[self.label2id[label["label"]], start_label_index] = 1
                        end_labels[self.label2id[label["label"]], end_label_index] = 1
                        span_labels[self.label2id[label["label"]], span_lookup[(start_label_index, end_label_index)]] = 1

                        annotation.append({
                            "start": start_label_index,
                            "end": end_label_index,
                            "label": label["label"]
                        })

            annotations["ner"].append(annotation)
            annotations["start_labels"].append(start_labels)
            annotations["end_labels"].append(end_labels)
            annotations["span_labels"].append(span_labels)
            annotations["valid_start_mask"].append(valid_start_mask)
            annotations["valid_end_mask"].append(valid_end_mask)
            annotations["valid_span_mask"].append(valid_span_mask)
            annotations["span_subword_indices"].append(span_subword_indices)
            annotations["span_lengths"].append(span_lengths)

        annotations["start_labels"] = torch.stack(annotations["start_labels"], dim=0)
        annotations["end_labels"] = torch.stack(annotations["end_labels"], dim=0)
        annotations["span_labels"] = torch.stack(annotations["span_labels"], dim=0)
        annotations["valid_start_mask"] = torch.stack(annotations["valid_start_mask"], dim=0)
        annotations["valid_end_mask"] = torch.stack(annotations["valid_end_mask"], dim=0)
        annotations["valid_span_mask"] = torch.stack(annotations["valid_span_mask"], dim=0)
        annotations["span_subword_indices"] = torch.stack(annotations["span_subword_indices"], dim=0)
        annotations["span_lengths"] = torch.stack(annotations["span_lengths"], dim=0)

        token_encoder_inputs = {
            "input_ids": token_encodings["input_ids"],
            "attention_mask": token_encodings["attention_mask"]
        }
        if "token_type_ids" in token_encodings:
            token_encoder_inputs["token_type_ids"] = token_encodings["token_type_ids"]
        
        type_encoder_inputs = {
            "input_ids": self.type_input_ids,
            "attention_mask": self.type_attention_mask
        }
        if self.type_token_type_ids is not None:
            type_encoder_inputs["token_type_ids"] = self.type_token_type_ids

        batch = {
            "token_encoder_inputs": token_encoder_inputs,
            "type_encoder_inputs": type_encoder_inputs,
            "labels": annotations,
            "id2label": {idx: label for label, idx in self.label2id.items()}
        }

        return batch