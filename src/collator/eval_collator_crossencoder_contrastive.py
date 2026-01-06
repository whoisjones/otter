import torch
import logging
from .masks import compressed_all_spans_mask_cross_encoder, compressed_subwords_mask_cross_encoder

logger = logging.getLogger(__name__)

class EvalCollatorContrastiveCrossEncoder:
    def __init__(self, tokenizer, label2id, max_seq_length=512, max_span_length=30, format='text', loss_masking='none', prediction_threshold='label_token'):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.format = format
        self.loss_masking = loss_masking
        
        if prediction_threshold not in ['label_token', 'cls']:
            raise ValueError(f"Invalid threshold token: {prediction_threshold}")
        if prediction_threshold == 'cls' and not self.tokenizer.cls_token == '[CLS]':
            logger.warning(f"CLS token not found in tokenizer. Using [SPAN_THRESHOLD] instead.")
            prediction_threshold = 'label_token'
        self.prediction_threshold = prediction_threshold
        self.threshold_token = '[SPAN_THRESHOLD]' if prediction_threshold == 'label_token' else '[CLS]'
        
        if self.format == 'text':
            label_prefix = "[LABEL] " + " [LABEL] ".join(self.label2id.keys()) + " [SEP]"
            if self.prediction_threshold == "label_token":
                label_prefix = self.threshold_token + " " + label_prefix
            self.label_offset = len(label_prefix)
            self.label_prefix = label_prefix
        elif self.format == 'tokens':
            label_list = [tok for label in self.label2id.keys() for tok in ('[LABEL]', label)] + ['[SEP]']
            if self.prediction_threshold == "label_token":
                label_list = [self.threshold_token] + label_list
            self.label_offset = len(label_list)
            self.label_prefix = label_list
        if loss_masking not in ['none', 'subwords']:
            raise ValueError(f"Invalid loss masking: {loss_masking}")

    def __call__(self, batch):
        if self.format == 'text':
            texts = [sample['text'] for sample in batch if len(sample['text']) > 2]
        elif self.format == 'tokens':
            texts = [sample["tokens"] for sample in batch if len(sample["tokens"]) > 2]
        else:
            raise ValueError(f"Invalid format: {self.format}")

        input_texts = [self.label_prefix + text for text in texts]
        
        token_encodings = self.tokenizer(
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

        threshold_token_subword_position = [next(i for i, input_id in enumerate(token_encodings['input_ids'][0]) if input_id == self.tokenizer.convert_tokens_to_ids(self.threshold_token))]
        label_token_subword_positions = [i for i, input_id in enumerate(token_encodings['input_ids'][0]) if input_id == self.tokenizer.convert_tokens_to_ids("[LABEL]")]

        annotations = {
            "ner": [],
            "span_lengths": [],
            "span_subword_indices": [],
            "valid_span_mask": []
        }

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = batch[i]["token_spans" if self.format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]

            if self.loss_masking == 'subwords':
                word_ids = token_encodings.word_ids(i)
                offsets = offset_mapping[i]
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_subwords_mask_cross_encoder(input_ids, word_ids, self.max_span_length, self.label_offset, offsets, has_threshold_token=True)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                offsets = offset_mapping[i]
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_all_spans_mask_cross_encoder(input_ids, sequence_ids, self.max_span_length, self.label_offset, offsets, has_threshold_token=True)

            span_subword_indices = torch.tensor(spans_idx)
            span_lengths = torch.tensor(span_lengths)
            valid_span_mask = torch.tensor([span_mask[:] for _ in range(len(self.label2id))])

            annotation = []

            for label in sample_labels:
                if self.format == 'text':
                    if offsets[text_start_index][0] <= label["start"] + self.label_offset and offsets[text_end_index][1] >= label["end"] + self.label_offset:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"] + self.label_offset:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"] + self.label_offset:
                            end_label_index -= 1
                        end_label_index += 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue
                        
                        start_label_index += 1
                        end_label_index += 1

                        annotation.append({
                            "start": start_label_index - text_start_index,
                            "end": end_label_index - text_start_index,
                            "label": label["label"]
                        })

                elif self.format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] + self.label_offset in word_ids and label["end"] - 1 + self.label_offset in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"] + self.label_offset:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"] + self.label_offset:
                            end_label_index -= 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_label_index += 1
                        end_label_index += 1

                        annotation.append({
                            "start": start_label_index - text_start_index,
                            "end": end_label_index - text_start_index,
                            "label": label["label"]
                        })

            annotations["ner"].append(annotation)
            annotations["span_subword_indices"].append(span_subword_indices)
            annotations["span_lengths"].append(span_lengths)
            annotations["valid_span_mask"].append(valid_span_mask)

        annotations["span_subword_indices"] = torch.stack(annotations["span_subword_indices"], dim=0)
        annotations["span_lengths"] = torch.stack(annotations["span_lengths"], dim=0)
        annotations["valid_span_mask"] = torch.stack(annotations["valid_span_mask"], dim=0)
        annotations["label_token_subword_positions"] = label_token_subword_positions
        annotations["threshold_token_subword_position"] = threshold_token_subword_position
        annotations["text_start_index"] = text_start_index

        token_encoder_inputs = {
            "input_ids": token_encodings["input_ids"],
            "attention_mask": token_encodings["attention_mask"]
        }
        if "token_type_ids" in token_encodings:
            token_encoder_inputs["token_type_ids"] = token_encodings["token_type_ids"]
        
        batch = {
            "token_encoder_inputs": token_encoder_inputs,
            "labels": annotations,
            "id2label": {idx: label for label, idx in self.label2id.items()}
        }

        return batch

