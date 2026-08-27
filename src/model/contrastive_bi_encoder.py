from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, MT5EncoderModel, PreTrainedModel

from ..config import SpanModelConfig
from ..loss import ContrastiveLoss
from .base import SpanModelOutput, mlp


class OtterContrastiveBiEncoderModel(PreTrainedModel):
    config_class = SpanModelConfig

    def __init__(self, config):
        super().__init__(config)
        token_config = AutoConfig.from_pretrained(config.token_encoder)
        type_config = AutoConfig.from_pretrained(config.type_encoder)

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size
        self.config.pruned_heads = getattr(token_config, "pruned_heads", {})

        self.type_linear = mlp(type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(
            token_config.hidden_size, config.linear_hidden_size, config.dropout
        )
        self.token_end_linear = mlp(
            token_config.hidden_size, config.linear_hidden_size, config.dropout
        )
        self.token_span_linear = mlp(
            config.linear_hidden_size * 2 + config.span_width_embedding_size,
            config.linear_hidden_size,
            config.dropout,
        )
        self.width_embedding = nn.Embedding(
            config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0
        )
        self.start_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / config.init_temperature)
        )
        self.end_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / config.init_temperature)
        )
        self.span_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / config.init_temperature)
        )
        if "mt5" in config.token_encoder:
            self.token_encoder = MT5EncoderModel(token_config)
        else:
            self.token_encoder = AutoModel.from_config(token_config)
        if "mt5" in config.type_encoder:
            self.type_encoder = MT5EncoderModel(type_config)
        else:
            self.type_encoder = AutoModel.from_config(type_config)
        self.post_init()

        if config.loss_fn != "contrastive":
            raise ValueError(f"Invalid loss function: {config.loss_fn}")
        self.loss_fn = ContrastiveLoss(tau=config.contrastive_tau)

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
        _, _, H = hidden_states.shape
        expanded_indices = span_indices.unsqueeze(2).expand(-1, -1, H)
        span_representations = torch.gather(hidden_states, 1, expanded_indices)
        return span_representations

    def forward(
        self,
        token_encoder_inputs: dict = None,
        type_encoder_inputs: dict = None,
        labels: dict = None,
    ):
        token_embeds = self.token_encoder(**token_encoder_inputs)
        type_embeds = self.type_encoder(**type_encoder_inputs)
        token_output = token_embeds.last_hidden_state

        if self.config.type_encoder_pooling == "mean":
            if type_encoder_inputs["attention_mask"] is not None:
                attention_mask_expanded = (
                    type_encoder_inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(type_embeds.last_hidden_state.size())
                    .float()
                )
                sum_embeddings = torch.sum(
                    type_embeds.last_hidden_state * attention_mask_expanded, dim=1
                )
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                type_output = sum_embeddings / sum_mask
            else:
                type_output = type_embeds.last_hidden_state.mean(dim=1)
        else:
            type_output = type_embeds.last_hidden_state[:, 0, :]

        B, S, H = token_output.size()
        C, _ = type_output.size()

        token_start_output = F.normalize(
            self.dropout(self.token_start_linear(token_output)), dim=-1
        )
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_output)), dim=-1)

        type_output = F.normalize(self.dropout(self.type_linear(type_output)), dim=-1)

        start_scores = self.start_logit_scale.exp() * torch.einsum(
            "BSH,CH->BCS", token_start_output, type_output
        )
        end_scores = self.end_logit_scale.exp() * torch.einsum(
            "BSH,CH->BCS", token_end_output, type_output
        )

        span_width_embeddings = self.width_embedding(labels["span_lengths"])

        span_hidden = torch.cat(
            [
                self.gather_spans(token_start_output, labels["span_subword_indices"][:, :, 0]),
                self.gather_spans(token_end_output, labels["span_subword_indices"][:, :, 1]),
                span_width_embeddings,
            ],
            dim=2,
        )

        token_span_output = F.normalize(self.dropout(self.token_span_linear(span_hidden)), dim=-1)

        span_scores = self.span_logit_scale.exp() * torch.einsum(
            "BSH,CH->BCS", token_span_output, type_output
        )

        if labels is not None and self.training:
            flat_start_scores = start_scores.reshape(B * C, S)
            flat_end_scores = end_scores.reshape(B * C, S)
            flat_span_scores = span_scores.reshape(B * C, span_scores.size(-1))
            start_negative_mask = labels["start_negative_mask"].reshape(B * C, S)
            end_negative_mask = labels["end_negative_mask"].reshape(B * C, S)
            span_negative_mask = labels["span_negative_mask"].reshape(B * C, span_scores.size(-1))

            start_threshold_loss = self.loss_fn(flat_start_scores, 0, start_negative_mask)
            end_threshold_loss = self.loss_fn(flat_end_scores, 0, end_negative_mask)
            span_threshold_loss = self.loss_fn(flat_span_scores, 0, span_negative_mask)

            threshold_loss = (
                self.config.start_loss_weight * start_threshold_loss
                + self.config.end_loss_weight * end_threshold_loss
                + self.config.span_loss_weight * span_threshold_loss
            )

            batch_indices, type_indices, start_indices, end_indices, span_indices = labels[
                "ner_indices"
            ]
            ner_start_mask, ner_end_mask, ner_span_mask = (
                labels["ner_start_mask"],
                labels["ner_end_mask"],
                labels["ner_span_mask"],
            )

            start_loss = self.loss_fn(
                start_scores[batch_indices, type_indices], start_indices, ner_start_mask
            )
            end_loss = self.loss_fn(
                end_scores[batch_indices, type_indices], end_indices, ner_end_mask
            )
            span_loss = self.loss_fn(
                span_scores[batch_indices, type_indices], span_indices, ner_span_mask
            )

            loss = (
                self.config.start_loss_weight * start_loss
                + self.config.end_loss_weight * end_loss
                + self.config.span_loss_weight * span_loss
            )

            total_loss = (
                self.config.contrastive_threshold_loss_weight * threshold_loss
                + self.config.contrastive_span_loss_weight * loss
            )

            return SpanModelOutput(
                loss=total_loss,
                start_logits=start_scores,
                end_logits=end_scores,
                span_logits=span_scores,
            )
        else:
            return SpanModelOutput(
                start_logits=start_scores, end_logits=end_scores, span_logits=span_scores
            )

    def save_pretrained(self, path: str, **kwargs):
        path = Path(path)
        token_tokenizer = kwargs.pop("token_tokenizer", None)
        type_tokenizer = kwargs.pop("type_tokenizer", None)
        super().save_pretrained(str(path), **kwargs)
        self.token_encoder.config.to_json_file(str(path / "token_encoder_config.json"))
        self.type_encoder.config.to_json_file(str(path / "type_encoder_config.json"))
        if token_tokenizer is not None:
            token_tokenizer.save_pretrained(str(path / "token_tokenizer"))
        if type_tokenizer is not None:
            type_tokenizer.save_pretrained(str(path / "type_tokenizer"))

    @classmethod
    def from_pretrained(cls, path: str, *model_args, config=None, **kwargs):
        path = Path(path)

        # New format: model.safetensors
        if (path / "model.safetensors").exists():
            span_cfg = SpanModelConfig.from_pretrained(str(path))
            model = cls(span_cfg)
            from safetensors.torch import load_file

            state_dict = load_file(str(path / "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
            return model

        # Legacy format: token_encoder/ + type_encoder/ dirs + model.pt
        span_cfg = SpanModelConfig.from_pretrained(str(path))
        model = cls(span_cfg)
        token_encoder_dir = path / "token_encoder"
        type_encoder_dir = path / "type_encoder"
        if token_encoder_dir.exists():
            if "mt5" in span_cfg.token_encoder:
                model.token_encoder = MT5EncoderModel.from_pretrained(str(token_encoder_dir))
            else:
                model.token_encoder = AutoModel.from_pretrained(str(token_encoder_dir))
        if type_encoder_dir.exists():
            if "mt5" in span_cfg.type_encoder:
                model.type_encoder = MT5EncoderModel.from_pretrained(str(type_encoder_dir))
            else:
                model.type_encoder = AutoModel.from_pretrained(str(type_encoder_dir))
        weights_file = path / "model.pt"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location="cpu")
            model_keys = set(model.state_dict().keys())
            model.load_state_dict(
                {k: v for k, v in state_dict.items() if k in model_keys}, strict=False
            )
        return model

    def gradient_checkpointing_enable(self):
        self.token_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()
