"""Otter: span-based multilingual NER with open (zero-shot) entity types.

Two architectures share this file:

* ``OtterBiEncoderModel``  -- the text and the entity type labels go through two
  separate encoders, and spans are scored against label embeddings. Label
  embeddings can be cached, so this is the cheaper option when the label set is
  fixed across many inputs.
* ``OtterCrossEncoderModel`` -- the labels are prepended to the text as a
  ``[LABEL] <type> ... [SEP] `` prefix and a single encoder sees both, so labels
  and text attend to each other. More accurate, more expensive.

Both expose :meth:`predict`, which takes raw strings plus a list of entity types
and returns character-level spans.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_otter import (OtterBiEncoderConfig,
                                  OtterCrossEncoderConfig, OtterConfig)
from .loss import BCELoss, DiceFocalLoss, DiceLoss, FocalLoss
from .masks import (compressed_all_spans_mask,
                    compressed_all_spans_mask_cross_encoder,
                    first_text_token_index)
from .metrics import compute_span_predictions


@dataclass
class SpanModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None


def mlp(input_size, output_size, dropout):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(output_size, output_size),
    )


def build_loss(config):
    if config.loss_fn == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    if config.loss_fn == "bce":
        return BCELoss()
    if config.loss_fn == "dice":
        return DiceLoss(smooth=getattr(config, "dice_smooth", 1.0))
    if config.loss_fn == "dice_focal":
        return DiceFocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            dice_weight=getattr(config, "dice_weight", 0.5),
            focal_weight=getattr(config, "focal_weight", 0.5),
            smooth=getattr(config, "dice_smooth", 1.0),
        )
    raise ValueError(f"Invalid loss function: {config.loss_fn}")


def _encoder_config(config, which):
    """Embedded sub-config, falling back to the base encoder on the Hub."""
    sub = getattr(config, f"{which}_config", None)
    if sub is not None:
        return sub
    return AutoConfig.from_pretrained(getattr(config, which))


def _weighted_span_loss(model, start_scores, end_scores, span_scores, labels):
    config = model.config

    def _pos_weight(name, scores):
        value = getattr(config, name, None)
        if value is None:
            return None
        return torch.tensor(value, device=scores.device, dtype=scores.dtype)

    start_loss = model.loss_fn(
        start_scores, labels["start_labels"],
        mask=labels["valid_start_mask"], pos_weight=_pos_weight("bce_start_pos_weight", start_scores),
    )
    end_loss = model.loss_fn(
        end_scores, labels["end_labels"],
        mask=labels["valid_end_mask"], pos_weight=_pos_weight("bce_end_pos_weight", end_scores),
    )
    span_loss = model.loss_fn(
        span_scores, labels["span_labels"],
        mask=labels["valid_span_mask"], pos_weight=_pos_weight("bce_span_pos_weight", span_scores),
    )
    return (
        config.start_loss_weight * start_loss
        + config.end_loss_weight * end_loss
        + config.span_loss_weight * span_loss
    )


class OtterPreTrainedModel(PreTrainedModel):
    config_class = OtterConfig
    base_model_prefix = "otter"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gather_spans(self, hidden_states, span_indices):
        _, _, hidden = hidden_states.shape
        expanded_indices = span_indices.unsqueeze(2).expand(-1, -1, hidden)
        return torch.gather(hidden_states, 1, expanded_indices)

    # -- inference helpers ------------------------------------------------

    def _resolve_threshold(self, threshold):
        if threshold is None:
            return self.config.prediction_threshold
        return threshold

    @staticmethod
    def _char_span(text, offsets, start_token, end_token, char_shift):
        """Token span -> character span, trimmed of any leading whitespace.

        Tokenizers that fold the preceding space into a token (mmBERT, and BPE
        tokenizers generally) report a character start one position early, so the
        raw offset can point at the space before the entity -- or, for the very
        first token of the cross-encoder's text, one character into the prompt
        prefix, which is why the result is clamped at zero.
        """
        char_start = max(0, int(offsets[start_token][0]) - char_shift)
        char_end = max(char_start, int(offsets[end_token][1]) - char_shift)
        surface = text[char_start:char_end]
        stripped = surface.lstrip()
        char_start += len(surface) - len(stripped)
        surface = stripped.rstrip()
        char_end = char_start + len(surface)
        return char_start, char_end, surface

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        labels: Sequence[str],
        threshold: Optional[float] = None,
        batch_size: int = 8,
        max_seq_length: Optional[int] = None,
    ) -> List[List[dict]]:
        """Extract entities of the given types from raw text.

        Args:
            texts: A string, or a list of strings.
            labels: Entity type names, e.g. ``["person", "organization"]``. Any
                natural-language type works; the model is not restricted to a
                fixed label set.
            threshold: Score above which a span is kept. Defaults to the value
                calibrated for this checkpoint (``config.prediction_threshold``).
            batch_size: Number of texts encoded per forward pass.
            max_seq_length: Truncation length. Defaults to
                ``config.max_seq_length``.

        Returns:
            One list per input text, each holding dicts with ``text``, ``label``,
            ``start``, ``end`` (character offsets into the input) and ``score``.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = list(texts)
        labels = list(labels)
        if not labels:
            raise ValueError("`labels` must contain at least one entity type.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"`labels` must not repeat an entity type: {labels}")

        threshold = self._resolve_threshold(threshold)
        max_seq_length = max_seq_length or self.config.max_seq_length
        was_training = self.training
        self.eval()

        # Blank inputs have no text tokens to build spans over, so they never reach the
        # model; they still get an (empty) result so the output lines up with the input.
        results: List[List[dict]] = [[] for _ in texts]
        todo = [i for i, text in enumerate(texts) if text.strip()]
        try:
            with torch.no_grad():
                for start in range(0, len(todo), batch_size):
                    indices = todo[start:start + batch_size]
                    predicted = self._predict_batch(
                        [texts[i] for i in indices], labels, threshold, max_seq_length
                    )
                    for i, entities in zip(indices, predicted):
                        results[i] = entities
        finally:
            if was_training:
                self.train()

        return results[0] if single else results


class OtterBiEncoderModel(OtterPreTrainedModel):
    """Separate encoders for text and for entity type names."""

    config_class = OtterBiEncoderConfig

    def __init__(self, config, token_config=None, type_config=None):
        super().__init__(config)
        self.token_config = token_config or _encoder_config(config, "token_encoder")
        self.type_config = type_config or _encoder_config(config, "type_encoder")

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size

        self.type_linear = mlp(self.type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_end_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_span_linear = mlp(
            config.linear_hidden_size * 2 + config.span_width_embedding_size,
            config.linear_hidden_size,
            config.dropout,
        )
        self.fusion_linear = mlp(config.linear_hidden_size * 2, config.linear_hidden_size, config.dropout)
        self.width_embedding = nn.Embedding(
            config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0
        )
        self.start_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.token_encoder = AutoModel.from_config(self.token_config)
        self.type_encoder = AutoModel.from_config(self.type_config)
        self.post_init()

        self.loss_fn = build_loss(config)
        self._token_tokenizer = None
        self._type_tokenizer = None

    def gradient_checkpointing_enable(self, **kwargs):
        self.token_encoder.gradient_checkpointing_enable(**kwargs)
        self.type_encoder.gradient_checkpointing_enable(**kwargs)

    def forward(self, token_encoder_inputs: dict = None, type_encoder_inputs: dict = None, labels: dict = None):
        token_output = self.token_encoder(**token_encoder_inputs).last_hidden_state
        type_embeds = self.type_encoder(**type_encoder_inputs).last_hidden_state

        if self.config.type_encoder_pooling == "mean":
            attention_mask = type_encoder_inputs.get("attention_mask")
            if attention_mask is not None:
                expanded = attention_mask.unsqueeze(-1).expand(type_embeds.size()).float()
                type_output = torch.sum(type_embeds * expanded, dim=1) / torch.clamp(expanded.sum(dim=1), min=1e-9)
            else:
                type_output = type_embeds.mean(dim=1)
        else:
            type_output = type_embeds[:, 0, :]

        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_output)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_output)), dim=-1)
        type_output = F.normalize(self.dropout(self.type_linear(type_output)), dim=-1)

        start_scores = self.start_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_start_output, type_output)
        end_scores = self.end_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_end_output, type_output)

        span_hidden = torch.cat(
            [
                self.gather_spans(token_start_output, labels["span_subword_indices"][:, :, 0]),
                self.gather_spans(token_end_output, labels["span_subword_indices"][:, :, 1]),
                self.width_embedding(labels["span_lengths"]),
            ],
            dim=2,
        )
        token_span_output = F.normalize(self.dropout(self.token_span_linear(span_hidden)), dim=-1)
        span_scores = self.span_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_span_output, type_output)

        loss = None
        if labels is not None and self.training:
            loss = _weighted_span_loss(self, start_scores, end_scores, span_scores, labels)
        return SpanModelOutput(loss=loss, start_logits=start_scores, end_logits=end_scores, span_logits=span_scores)

    # -- inference --------------------------------------------------------

    @property
    def token_tokenizer(self):
        if self._token_tokenizer is None:
            self._token_tokenizer = _load_tokenizer(self, "token_tokenizer", self.config.token_encoder)
        return self._token_tokenizer

    @property
    def type_tokenizer(self):
        if self._type_tokenizer is None:
            self._type_tokenizer = _load_tokenizer(self, "type_tokenizer", self.config.type_encoder)
        return self._type_tokenizer

    def encode_labels(self, labels: Sequence[str]) -> dict:
        """Tokenize entity type names once so they can be reused across batches."""
        encoding = self.type_tokenizer(list(labels), padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoding.items()}

    def _predict_batch(self, texts, labels, threshold, max_seq_length):
        encodings = self.token_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encodings.pop("offset_mapping")

        span_indices, span_lengths, span_masks = [], [], []
        for i in range(len(texts)):
            _, _, _, _, span_mask, spans_idx, lengths = compressed_all_spans_mask(
                encodings["input_ids"][i], encodings.sequence_ids(i), self.config.max_span_length
            )
            span_indices.append(torch.tensor(spans_idx))
            span_lengths.append(torch.tensor(lengths))
            span_masks.append(torch.tensor([span_mask[:] for _ in labels]))

        model_labels = {
            "span_subword_indices": torch.stack(span_indices).to(self.device),
            "span_lengths": torch.stack(span_lengths).to(self.device),
            "valid_span_mask": torch.stack(span_masks).to(self.device),
        }
        token_inputs = {k: v.to(self.device) for k, v in encodings.items()}

        output = self(
            token_encoder_inputs=token_inputs,
            type_encoder_inputs=self.encode_labels(labels),
            labels=model_labels,
        )
        return _decode(
            self, output, model_labels, texts, offset_mapping, labels, threshold,
            token_offset=0, char_shift=0,
        )


class OtterCrossEncoderModel(OtterPreTrainedModel):
    """One encoder over ``[LABEL] <type> ... [SEP] <text>``."""

    config_class = OtterCrossEncoderConfig

    def __init__(self, config, token_config=None):
        super().__init__(config)
        self.token_config = token_config or _encoder_config(config, "token_encoder")

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size

        self.type_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_end_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_span_linear = mlp(
            config.linear_hidden_size * 2 + config.span_width_embedding_size,
            config.linear_hidden_size,
            config.dropout,
        )
        self.width_embedding = nn.Embedding(
            config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0
        )
        self.start_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.token_encoder = AutoModel.from_config(self.token_config)
        self.post_init()

        self.loss_fn = build_loss(config)
        self._tokenizer = None

    def gradient_checkpointing_enable(self, **kwargs):
        self.token_encoder.gradient_checkpointing_enable(**kwargs)

    def forward(self, token_encoder_inputs: dict = None, labels: dict = None, **kwargs):
        encoder_outputs = self.token_encoder(**token_encoder_inputs).last_hidden_state
        token_hidden = encoder_outputs[:, labels["text_start_index"]:, :]
        type_hidden = encoder_outputs[:, labels["label_token_subword_positions"], :]

        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_hidden)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_hidden)), dim=-1)
        type_output = F.normalize(self.dropout(self.type_linear(type_hidden)), dim=-1)

        start_scores = self.start_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_start_output, type_output)
        end_scores = self.end_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_end_output, type_output)

        span_hidden = torch.cat(
            [
                self.gather_spans(token_start_output, labels["span_subword_indices"][:, :, 0]),
                self.gather_spans(token_end_output, labels["span_subword_indices"][:, :, 1]),
                self.width_embedding(labels["span_lengths"]),
            ],
            dim=2,
        )
        token_span_output = F.normalize(self.dropout(self.token_span_linear(span_hidden)), dim=-1)
        span_scores = self.span_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_span_output, type_output)

        loss = None
        if labels is not None and self.training:
            loss = _weighted_span_loss(self, start_scores, end_scores, span_scores, labels)
        return SpanModelOutput(loss=loss, start_logits=start_scores, end_logits=end_scores, span_logits=span_scores)

    # -- inference --------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self, None, self.config.token_encoder)
        return self._tokenizer

    @staticmethod
    def build_prompt(labels: Sequence[str]) -> str:
        """The label prefix the model was trained with, e.g. ``[LABEL] person [SEP] ``."""
        return "[LABEL] " + " [LABEL] ".join(labels) + " [SEP] "

    def _predict_batch(self, texts, labels, threshold, max_seq_length):
        prefix = self.build_prompt(labels)
        label_offset = len(prefix)

        encodings = self.tokenizer(
            [prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encodings.pop("offset_mapping")

        label_token_id = self.tokenizer.convert_tokens_to_ids("[LABEL]")
        label_positions = [
            i for i, token_id in enumerate(encodings["input_ids"][0].tolist()) if token_id == label_token_id
        ]
        if len(label_positions) != len(labels):
            raise ValueError(
                f"Expected {len(labels)} [LABEL] markers in the prompt but found {len(label_positions)}. "
                "The label prefix was probably truncated -- reduce the number of entity types or raise "
                "`max_seq_length`."
            )

        # One index for the whole batch: the prefix is identical everywhere, but the
        # token straddling the prefix/text boundary is not, and a per-sample index
        # would give tensors of different widths that cannot be stacked.
        text_start_index = first_text_token_index(offset_mapping, label_offset)

        span_indices, span_lengths, span_masks = [], [], []
        for i in range(len(texts)):
            _, _, _, _, span_mask, spans_idx, lengths = compressed_all_spans_mask_cross_encoder(
                encodings["input_ids"][i],
                encodings.sequence_ids(i),
                self.config.max_span_length,
                label_offset,
                offset_mapping[i],
                text_start_index=text_start_index,
            )
            span_indices.append(torch.tensor(spans_idx))
            span_lengths.append(torch.tensor(lengths))
            span_masks.append(torch.tensor([span_mask[:] for _ in labels]))

        model_labels = {
            "span_subword_indices": torch.stack(span_indices).to(self.device),
            "span_lengths": torch.stack(span_lengths).to(self.device),
            "valid_span_mask": torch.stack(span_masks).to(self.device),
            "text_start_index": text_start_index,
            "label_token_subword_positions": label_positions,
        }
        token_inputs = {k: v.to(self.device) for k, v in encodings.items()}

        output = self(token_encoder_inputs=token_inputs, labels=model_labels)
        return _decode(
            self, output, model_labels, texts, offset_mapping, labels, threshold,
            token_offset=text_start_index, char_shift=label_offset,
        )


def _load_tokenizer(model, subfolder, fallback):
    """Prefer the tokenizer shipped with the checkpoint; fall back to the base encoder."""
    source = model.name_or_path
    if source:
        try:
            kwargs = {"subfolder": subfolder} if subfolder else {}
            # trust_remote_code: the checkpoint's own config declares custom classes,
            # and the user already opted into them to build this model.
            return AutoTokenizer.from_pretrained(source, trust_remote_code=True, **kwargs)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(fallback)


def _decode(model, output, model_labels, texts, offset_mapping, labels, threshold, token_offset, char_shift):
    """Turn span logits into character-level entity dicts."""
    predictions = compute_span_predictions(
        span_logits=output.span_logits.detach().float().cpu().numpy(),
        span_mask=model_labels["valid_span_mask"].cpu().numpy(),
        span_mapping=model_labels["span_subword_indices"].cpu().numpy(),
        id2label={idx: label for idx, label in enumerate(labels)},
        threshold=threshold,
    )

    decoded = []
    for i, text in enumerate(texts):
        offsets = offset_mapping[i]
        entities = []
        for span in predictions[i]:
            start_token = span["start"] + token_offset
            end_token = span["end"] + token_offset
            char_start, char_end, surface = model._char_span(text, offsets, start_token, end_token, char_shift)
            if not surface:
                continue
            entities.append({
                "text": surface,
                "label": span["label"],
                "start": char_start,
                "end": char_end,
                "score": float(span["confidence"]),
            })
        entities.sort(key=lambda e: (e["start"], e["end"]))
        decoded.append(entities)
    return decoded
