import torch


class InBatchDataCollator:
    def __init__(self, token_encoder_tokenizer, type_encoder_tokenizer, max_seq_length=512, max_span_length=30, format='text'):
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
        unique_types = sorted(unique_types)
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
            "ner": [],
            "start_labels": [],
            "end_labels": [],
            "span_labels": [],
            "start_loss_mask": [],
            "end_loss_mask": [],
            "span_loss_mask": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]

            input_ids = token_encodings['input_ids'][i]
            word_ids = token_encodings.word_ids(i)
            
            text_start_index = 0
            while word_ids[text_start_index] == None:
                text_start_index += 1

            text_end_index = len(input_ids) - 1
            while word_ids[text_end_index] == None:
                text_end_index -= 1

            # For each offset, set start_mask to 1 if it's the first occurrence of a word_id, and end_mask to 1 if it's the last occurrence of that word_id
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

            start_loss_mask = torch.tensor([start_mask[:] for _ in unique_types])
            end_loss_mask = torch.tensor([end_mask[:] for _ in unique_types])
            span_loss_mask = torch.tensor([[x[:] for x in span_mask] for _ in unique_types])

            start_labels = torch.zeros(len(unique_types), len(input_ids))
            end_labels = torch.zeros(len(unique_types), len(input_ids))
            span_labels = torch.zeros(len(unique_types), len(input_ids), len(input_ids))

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

                        start_labels[type2id_batch[label["label"]], start_label_index] = 1
                        end_labels[type2id_batch[label["label"]], end_label_index] = 1
                        span_labels[type2id_batch[label["label"]], start_label_index, end_label_index] = 1

                        annotation.append({
                            "start": start_label_index,
                            "end": end_label_index,
                            "label": type2id_batch[label["label"]]
                        })

                elif self.format == 'tokens':
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

        annotations["start_labels"] = torch.stack(annotations["start_labels"], dim=0)
        annotations["end_labels"] = torch.stack(annotations["end_labels"], dim=0)
        annotations["start_loss_mask"] = torch.stack(annotations["start_loss_mask"], dim=0)
        annotations["end_loss_mask"] = torch.stack(annotations["end_loss_mask"], dim=0)
        annotations["span_labels"] = torch.stack(annotations["span_labels"], dim=0)
        annotations["span_loss_mask"] = torch.stack(annotations["span_loss_mask"], dim=0)
        annotations["start_loss_weight"] = (annotations["start_loss_mask"].sum() / annotations["start_labels"].sum())
        annotations["end_loss_weight"] = (annotations["end_loss_mask"].sum() / annotations["end_labels"].sum())
        annotations["span_loss_weight"] = (annotations["span_loss_mask"].sum() / annotations["span_labels"].sum())

        batch = {
            "token_input_ids": token_encodings["input_ids"],
            "token_attention_mask": token_encodings["attention_mask"],
            "type_input_ids": type_encodings["input_ids"],
            "type_attention_mask": type_encodings["attention_mask"],
            "labels": annotations
        }
        
        if "type_ids" in token_encodings:
            batch["token_type_ids"] = token_encodings["type_ids"]
        if "type_ids" in type_encodings:
            batch["type_type_ids"] = type_encodings["type_ids"]

        return batch

class AllLabelsDataCollator:
    def __init__(self, tokenizer, type_encodings, label2id, max_seq_length=512, max_span_length=30, format='text'):
        self.tokenizer = tokenizer
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
        
        token_encodings = self.tokenizer(
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
            "start_loss_mask": [],
            "end_loss_mask": [],
            "span_loss_mask": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]

            input_ids = token_encodings['input_ids'][i]
            word_ids = token_encodings.word_ids(i)
            
            text_start_index = 0
            while word_ids[text_start_index] == None:
                text_start_index += 1

            text_end_index = len(input_ids) - 1
            while word_ids[text_end_index] == None:
                text_end_index -= 1

            # For each offset, set start_mask to 1 if it's the first occurrence of a word_id, and end_mask to 1 if it's the last occurrence of that word_id
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

            start_loss_mask = torch.tensor([start_mask[:] for _ in range(len(self.label2id))])
            end_loss_mask = torch.tensor([end_mask[:] for _ in range(len(self.label2id))])
            span_loss_mask = torch.tensor([[x[:] for x in span_mask] for _ in range(len(self.label2id))])

            start_labels = torch.zeros(len(self.label2id), len(input_ids))
            end_labels = torch.zeros(len(self.label2id), len(input_ids))
            span_labels = torch.zeros(len(self.label2id), len(input_ids), len(input_ids))

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

                        start_labels[self.label2id[label["label"]], start_label_index] = 1
                        end_labels[self.label2id[label["label"]], end_label_index] = 1
                        span_labels[self.label2id[label["label"]], start_label_index, end_label_index] = 1

                        annotation.append({
                            "start": start_label_index,
                            "end": end_label_index,
                            "label": label["label"]
                        })

                elif self.format == 'tokens':
                    if label["start"] in word_ids and label["end"] - 1 in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"]:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"]:
                            end_label_index -= 1

                        start_labels[self.label2id[label["label"]], start_label_index] = 1
                        end_labels[self.label2id[label["label"]], end_label_index] = 1
                        span_labels[self.label2id[label["label"]], start_label_index, end_label_index] = 1

                        annotation.append({
                            "start": start_label_index,
                            "end": end_label_index,
                            "label": label["label"]
                        })

            annotations["ner"].append(annotation)
            annotations["start_labels"].append(start_labels)
            annotations["end_labels"].append(end_labels)
            annotations["span_labels"].append(span_labels)
            annotations["start_loss_mask"].append(start_loss_mask)
            annotations["end_loss_mask"].append(end_loss_mask)
            annotations["span_loss_mask"].append(span_loss_mask)

        annotations["start_labels"] = torch.stack(annotations["start_labels"], dim=0)
        annotations["end_labels"] = torch.stack(annotations["end_labels"], dim=0)
        annotations["start_loss_mask"] = torch.stack(annotations["start_loss_mask"], dim=0)
        annotations["end_loss_mask"] = torch.stack(annotations["end_loss_mask"], dim=0)
        annotations["span_labels"] = torch.stack(annotations["span_labels"], dim=0)
        annotations["span_loss_mask"] = torch.stack(annotations["span_loss_mask"], dim=0)

        batch = {
            "token_input_ids": token_encodings["input_ids"],
            "token_attention_mask": token_encodings["attention_mask"],
            "type_input_ids": self.type_input_ids,
            "type_attention_mask": self.type_attention_mask,
            "labels": annotations,
            "id2label": {idx: label for label, idx in self.label2id.items()}
        }

        if "token_type_ids" in token_encodings:
            batch["token_token_type_ids"] = token_encodings["token_type_ids"]
        if self.type_token_type_ids is not None:
            batch["type_token_type_ids"] = self.type_token_type_ids

        return batch