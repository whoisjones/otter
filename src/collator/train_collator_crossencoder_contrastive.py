import random
import torch
import logging
from .masks import compressed_all_spans_mask_cross_encoder, compressed_subwords_mask_cross_encoder

logger = logging.getLogger(__name__)

class TrainCollatorContrastiveCrossEncoder:
    def __init__(self, token_encoder_tokenizer, max_seq_length=512, max_span_length=30, format='text', loss_masking='none', prediction_threshold='label_token'):
        self.token_encoder_tokenizer = token_encoder_tokenizer
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.format = format
        self.loss_masking = loss_masking
        if prediction_threshold not in ['label_token', 'cls']:
            raise ValueError(f"Invalid threshold token: {prediction_threshold}")
        if prediction_threshold == 'cls' and not self.token_encoder_tokenizer.cls_token == '[CLS]':
            logger.warning(f"CLS token not found in tokenizer. Using [SPAN_THRESHOLD] instead.")
            prediction_threshold = 'label_token'
        self.prediction_threshold = prediction_threshold
        self.threshold_token = '[SPAN_THRESHOLD]' if prediction_threshold == 'label_token' else '[CLS]'
        if loss_masking not in ['none', 'subwords']:
            raise ValueError(f"Invalid loss masking: {loss_masking}")

    def __call__(self, batch):
        if self.format == 'text':
            texts = [sample['text'] for sample in batch]
        elif self.format == 'tokens':
            texts = [sample["tokens"] for sample in batch]
        else:
            raise ValueError(f"Invalid format: {self.format}")

        unique_types = []
        for sample in batch:
            for span in sample["token_spans" if self.format == 'tokens' else "char_spans"]:
                if span["label"] not in unique_types:
                    unique_types.append(span["label"])
        random.shuffle(unique_types)
        type2id_batch = {entity_type: idx for idx, entity_type in enumerate(unique_types)}

        if not unique_types:
            return {}

        if self.format == 'text':
            label_text = "[LABEL] " + " [LABEL] ".join(unique_types) + " [SEP]"
            if self.prediction_threshold == "label_token":
                label_text = self.threshold_token + " " + label_text
            label_offset = len(label_text)
            input_texts = [label_text + text for text in texts]
        elif self.format == 'tokens':
            label_list = [tok for label in unique_types for tok in ('[LABEL]', label)] + ['[SEP]']
            if self.prediction_threshold == "label_token":
                label_list = [self.threshold_token] + label_list
            label_offset = len(label_list)
            input_texts = [label_list + text for text in texts]
        
        token_encodings = self.token_encoder_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True if self.format == 'text' else False,
            is_split_into_words=True if self.format == 'tokens' else False
        )

        if self.format == 'text':
            offset_mapping = token_encodings.pop("offset_mapping")

        threshold_token_subword_position = [next(i for i, input_id in enumerate(token_encodings['input_ids'][0]) if input_id == self.token_encoder_tokenizer.convert_tokens_to_ids(self.threshold_token))]
        label_token_subword_positions = [i for i, input_id in enumerate(token_encodings['input_ids'][0]) if input_id == self.token_encoder_tokenizer.convert_tokens_to_ids("[LABEL]")]

        annotations = {
            "ner_indices": [[], [], [], [], []],
            "ner_start_mask": [],
            "ner_end_mask": [],
            "ner_span_mask": [],
            "start_negative_mask": [],
            "end_negative_mask": [],
            "span_negative_mask": [],
            "span_lengths": [],
            "span_subword_indices": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]

            if self.loss_masking == 'subwords':
                word_ids = token_encodings.word_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_subwords_mask_cross_encoder(input_ids, word_ids, self.max_span_length, label_offset, has_threshold_token=True)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                offsets = offset_mapping[i]
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_all_spans_mask_cross_encoder(input_ids, sequence_ids, self.max_span_length, label_offset, offsets, has_threshold_token=True)

            span_lookup = {span: idx for idx, span in enumerate(spans_idx)}
            span_subword_indices = torch.tensor(spans_idx)
            span_lengths = torch.tensor(span_lengths)

            start_negative_mask = torch.tensor([start_mask[:] for _ in unique_types])
            end_negative_mask = torch.tensor([end_mask[:] for _ in unique_types])
            span_negative_mask = torch.tensor([span_mask[:] for _ in unique_types])
            start_negative_mask[:, 0] = 1
            end_negative_mask[:, 0] = 1
            span_negative_mask[:, 0] = 1

            for label in sample_labels:
                if self.format == 'text':
                    if offsets[text_start_index][0] <= label["start"] + label_offset and offsets[text_end_index][1] >= label["end"] + label_offset:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"] + label_offset:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"] + label_offset:
                            end_label_index -= 1
                        end_label_index += 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_label_index += 1
                        end_label_index += 1

                        start_negative_mask[type2id_batch[label["label"]], (start_label_index - text_start_index)] = 0
                        end_negative_mask[type2id_batch[label["label"]], (end_label_index - text_start_index)] = 0
                        span_negative_mask[type2id_batch[label["label"]], span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)]] = 0

                        annotations["ner_indices"][0].append(i)
                        annotations["ner_indices"][1].append(type2id_batch[label["label"]])
                        annotations["ner_indices"][2].append(start_label_index - text_start_index)
                        annotations["ner_indices"][3].append(end_label_index - text_start_index)
                        annotations["ner_indices"][4].append(span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)])

                elif self.format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] + label_offset in word_ids and label["end"] - 1 + label_offset in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"] + label_offset:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"] + label_offset:
                            end_label_index -= 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_label_index += 1
                        end_label_index += 1

                        start_negative_mask[type2id_batch[label["label"]], (start_label_index - text_start_index)] = 0
                        end_negative_mask[type2id_batch[label["label"]], (end_label_index - text_start_index)] = 0
                        span_negative_mask[type2id_batch[label["label"]], span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)]] = 0
                        
                        annotations["ner_indices"][0].append(i)
                        annotations["ner_indices"][1].append(type2id_batch[label["label"]])
                        annotations["ner_indices"][2].append(start_label_index - text_start_index)
                        annotations["ner_indices"][3].append(end_label_index - text_start_index)
                        annotations["ner_indices"][4].append(span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)])

            annotations["start_negative_mask"].append(start_negative_mask)
            annotations["end_negative_mask"].append(end_negative_mask)
            annotations["span_negative_mask"].append(span_negative_mask)
            annotations["span_subword_indices"].append(span_subword_indices)
            annotations["span_lengths"].append(span_lengths)

        annotations['start_negative_mask'] = torch.stack(annotations['start_negative_mask'], dim=0)
        annotations['end_negative_mask'] = torch.stack(annotations['end_negative_mask'], dim=0)
        annotations['span_negative_mask'] = torch.stack(annotations['span_negative_mask'], dim=0)
        annotations["span_subword_indices"] = torch.stack(annotations["span_subword_indices"], dim=0)
        annotations["span_lengths"] = torch.stack(annotations["span_lengths"], dim=0)
        annotations["label_token_subword_positions"] = label_token_subword_positions
        annotations["threshold_token_subword_position"] = threshold_token_subword_position
        annotations["text_start_index"] = text_start_index

        for i in range(len(annotations["ner_indices"][0])):
            batch_index = annotations["ner_indices"][0][i]
            type_index = annotations["ner_indices"][1][i]
            start_label_index = annotations["ner_indices"][2][i]
            end_label_index = annotations["ner_indices"][3][i]
            span_index = annotations["ner_indices"][4][i]
            
            ner_start_mask = annotations['start_negative_mask'][batch_index, type_index].detach().clone()
            ner_start_mask[start_label_index] = 1
            annotations["ner_start_mask"].append(ner_start_mask)
            ner_end_mask = annotations['end_negative_mask'][batch_index,type_index].detach().clone()
            ner_end_mask[end_label_index] = 1
            annotations["ner_end_mask"].append(ner_end_mask)
            ner_span_mask = annotations['span_negative_mask'][batch_index, type_index].detach().clone()
            ner_span_mask[span_index] = 1
            annotations["ner_span_mask"].append(ner_span_mask)

        annotations['ner_start_mask'] = torch.stack(annotations['ner_start_mask'], dim=0)
        annotations['ner_end_mask'] = torch.stack(annotations['ner_end_mask'], dim=0)
        annotations['ner_span_mask'] = torch.stack(annotations['ner_span_mask'], dim=0)

        token_encoder_inputs = {
            "input_ids": token_encodings["input_ids"],
            "attention_mask": token_encodings["attention_mask"]
        }
        if "token_type_ids" in token_encodings:
            token_encoder_inputs["token_type_ids"] = token_encodings["token_type_ids"]
        
        batch = {
            "token_encoder_inputs": token_encoder_inputs,
            "labels": annotations
        }

        return batch

