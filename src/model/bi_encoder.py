import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (AutoConfig, AutoModel, MT5EncoderModel,
                          PretrainedConfig, PreTrainedModel)

from ..config import SpanModelConfig
from ..loss import BCELoss, FocalLoss, DiceLoss, DiceFocalLoss
from .base import SpanModelOutput, mlp


class OtterBiEncoderModel(PreTrainedModel):
    """Dual encoder with span marker module."""

    config_class = SpanModelConfig

    def __init__(self, config, token_config=None, type_config=None):
        super().__init__(config)
        self.config = config
        self.token_config = token_config
        self.type_config = type_config

        if self.token_config is None:
            self.token_config = AutoConfig.from_pretrained(config.token_encoder)
        if self.type_config is None:
            self.type_config = AutoConfig.from_pretrained(config.type_encoder)

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size
        self.config.pruned_heads = getattr(self.token_config, "pruned_heads", {})

        self.type_linear = mlp(self.type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_end_linear = mlp(self.token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_span_linear = mlp(config.linear_hidden_size * 2 + config.span_width_embedding_size, config.linear_hidden_size, config.dropout)
        self.fusion_linear = mlp(config.linear_hidden_size * 2, config.linear_hidden_size, config.dropout)
        self.width_embedding = nn.Embedding(config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0)
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        if "mt5" in config.token_encoder:
            self.token_encoder = MT5EncoderModel(self.token_config)
        else:
            self.token_encoder = AutoModel.from_config(self.token_config)
        if "mt5" in config.type_encoder:
            self.type_encoder = MT5EncoderModel(self.type_config)
        else:
            self.type_encoder = AutoModel.from_config(self.type_config)
        self.post_init()

        if config.loss_fn == "focal":
            self.loss_fn = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        elif config.loss_fn == "bce":
            self.loss_fn = BCELoss()
        elif config.loss_fn == "dice":
            self.loss_fn = DiceLoss(smooth=getattr(config, "dice_smooth", 1.0))
        elif config.loss_fn == "dice_focal":
            self.loss_fn = DiceFocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                dice_weight=getattr(config, "dice_weight", 0.5),
                focal_weight=getattr(config, "focal_weight", 0.5),
                smooth=getattr(config, "dice_smooth", 1.0),
            )
        else:
            raise ValueError(f"Invalid loss function: {config.loss_fn}")

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
        labels: dict = None
    ):
        token_embeds = self.token_encoder(**token_encoder_inputs)
        type_embeds = self.type_encoder(**type_encoder_inputs)
        token_output = token_embeds.last_hidden_state
        
        if self.config.type_encoder_pooling == "mean":
            if type_encoder_inputs["attention_mask"] is not None:
                attention_mask_expanded = type_encoder_inputs["attention_mask"].unsqueeze(-1).expand(type_embeds.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(type_embeds.last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                type_output = sum_embeddings / sum_mask
            else:
                type_output = type_embeds.last_hidden_state.mean(dim=1)
        else:
            type_output = type_embeds.last_hidden_state[:, 0, :]

        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_output)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_output)), dim=-1)

        type_output = F.normalize(self.dropout(self.type_linear(type_output)), dim=-1)

        start_scores = self.start_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_start_output, type_output)
        end_scores = self.end_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_end_output, type_output)
        
        span_width_embeddings = self.width_embedding(labels["span_lengths"])

        span_hidden = torch.cat(
            [
                self.gather_spans(token_start_output, labels["span_subword_indices"][:, :, 0]),
                self.gather_spans(token_end_output, labels["span_subword_indices"][:, :, 1]),
                span_width_embeddings,
            ],
            dim=2
        )

        token_span_output = F.normalize(self.dropout(self.token_span_linear(span_hidden)), dim=-1)
        span_scores = self.span_logit_scale.exp() * torch.einsum("BSH,CH->BCS", token_span_output, type_output)

        if labels is not None and self.training:
            start_pos_weight = None
            if self.config.bce_start_pos_weight is not None:
                start_pos_weight = torch.tensor(self.config.bce_start_pos_weight, device=start_scores.device, dtype=start_scores.dtype)
            
            end_pos_weight = None
            if self.config.bce_end_pos_weight is not None:
                end_pos_weight = torch.tensor(self.config.bce_end_pos_weight, device=end_scores.device, dtype=end_scores.dtype)
            
            span_pos_weight = None
            if self.config.bce_span_pos_weight is not None:
                span_pos_weight = torch.tensor(self.config.bce_span_pos_weight, device=span_scores.device, dtype=span_scores.dtype)
            
            start_loss = self.loss_fn(
                start_scores, 
                labels["start_labels"],
                mask=labels["valid_start_mask"],
                pos_weight=start_pos_weight
            )

            end_loss = self.loss_fn(
                end_scores,
                labels["end_labels"],
                mask=labels["valid_end_mask"],
                pos_weight=end_pos_weight
            )

            span_loss = self.loss_fn(
                span_scores, 
                labels["span_labels"],
                mask=labels["valid_span_mask"],
                pos_weight=span_pos_weight
            )
            loss = self.config.start_loss_weight * start_loss + self.config.end_loss_weight * end_loss + self.config.span_loss_weight * span_loss
            return SpanModelOutput(loss=loss, start_logits=start_scores, end_logits=end_scores, span_logits=span_scores)
        else:
            return SpanModelOutput(start_logits=start_scores, end_logits=end_scores, span_logits=span_scores)
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        if not isinstance(save_directory, Path):
            save_directory = Path(save_directory)

        token_tokenizer = kwargs.pop("token_tokenizer", None)
        type_tokenizer = kwargs.pop("type_tokenizer", None)

        super().save_pretrained(save_directory, **kwargs)

        self.token_encoder.config.to_json_file(str(save_directory / "token_encoder_config.json"))
        self.type_encoder.config.to_json_file(str(save_directory / "type_encoder_config.json"))

        if token_tokenizer is not None:
            token_tokenizer.save_pretrained(str(save_directory / "token_tokenizer"))
        if type_tokenizer is not None:
            type_tokenizer.save_pretrained(str(save_directory / "type_tokenizer"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        **kwargs,
    ):
        base_path = Path(pretrained_model_name_or_path)

        # New format: model.safetensors
        model_safetensors = base_path / "model.safetensors"
        token_encoder_config_path = base_path / "token_encoder_config.json"
        type_encoder_config_path = base_path / "type_encoder_config.json"

        if model_safetensors.exists():
            span_cfg = SpanModelConfig.from_pretrained(str(base_path))
            token_config = AutoConfig.from_pretrained(token_encoder_config_path) if token_encoder_config_path.exists() else AutoConfig.from_pretrained(span_cfg.token_encoder)
            type_config = AutoConfig.from_pretrained(type_encoder_config_path) if type_encoder_config_path.exists() else AutoConfig.from_pretrained(span_cfg.type_encoder)
            model = cls(span_cfg, token_config=token_config, type_config=type_config)
            from safetensors.torch import load_file
            state_dict = load_file(str(model_safetensors))
            model.load_state_dict(state_dict, strict=False)
            return model

        # Legacy format: token_encoder/ + type_encoder/ dirs + model.pt (span head only)
        span_cfg = SpanModelConfig.from_pretrained(str(base_path))
        token_encoder_dir = base_path / "token_encoder"
        type_encoder_dir = base_path / "type_encoder"
        token_config = AutoConfig.from_pretrained(str(token_encoder_dir) if token_encoder_dir.exists() else span_cfg.token_encoder)
        type_config = AutoConfig.from_pretrained(str(type_encoder_dir) if type_encoder_dir.exists() else span_cfg.type_encoder)
        model = cls(span_cfg, token_config=token_config, type_config=type_config)
        if token_encoder_dir.exists():
            if "mt5" in span_cfg.token_encoder:
                model.token_encoder = MT5EncoderModel.from_pretrained(str(token_encoder_dir), config=token_config)
            else:
                model.token_encoder = AutoModel.from_pretrained(str(token_encoder_dir), config=token_config)
        if type_encoder_dir.exists():
            if "mt5" in span_cfg.type_encoder:
                model.type_encoder = MT5EncoderModel.from_pretrained(str(type_encoder_dir), config=type_config)
            else:
                model.type_encoder = AutoModel.from_pretrained(str(type_encoder_dir), config=type_config)
        weights_file = base_path / "span_model_weights.pt"
        if not weights_file.exists():
            weights_file = base_path / "model.pt"
        if weights_file.exists():
            span_state = torch.load(weights_file, map_location="cpu")
            model_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in span_state.items() if k in model_keys}
            model.load_state_dict(filtered, strict=False)
        return model

    def gradient_checkpointing_enable(self):
        self.token_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

