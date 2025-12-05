import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig, PreTrainedModel, MT5EncoderModel
from pathlib import Path

from .base import SpanModelOutput, mlp
from ..config import SpanModelConfig
from ..loss import BCELoss, FocalLoss, JGMakerLoss, TokenizationAwareLoss

class SpanModel(PreTrainedModel):
    """Dual encoder with span marker module."""

    def __init__(self, config):
        super().__init__(config)
        token_config = AutoConfig.from_pretrained(config.token_encoder)
        type_config = AutoConfig.from_pretrained(config.type_encoder)

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size
        self.config.pruned_heads = token_config.pruned_heads

        self.type_start_linear = mlp(type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.type_end_linear = mlp(type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.type_span_linear = mlp(type_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_end_linear = mlp(token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_span_linear = mlp(token_config.hidden_size * 2 + config.span_width_embedding_size, config.linear_hidden_size, config.dropout)
        self.width_embedding = nn.Embedding(config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0)
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.post_init()

        if "mt5" in config.token_encoder:
            self.token_encoder = MT5EncoderModel.from_pretrained(config.token_encoder, config=token_config)
        else:
            self.token_encoder = AutoModel.from_pretrained(config.token_encoder, config=token_config)
        if "mt5" in config.type_encoder:
            self.type_encoder = MT5EncoderModel.from_pretrained(config.type_encoder, config=type_config)
        else:
            self.type_encoder = AutoModel.from_pretrained(config.type_encoder, config=type_config)

        if config.loss_fn == "focal":
            self.loss_fn = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        elif config.loss_fn == "jgmaker":
            self.loss_fn = JGMakerLoss(total_steps=3000)
        elif config.loss_fn == "bce":
            self.loss_fn = BCELoss()
        elif config.loss_fn == "tokenization_aware":
            self.loss_fn = TokenizationAwareLoss()
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

        batch_size, seq_length, _ = token_output.size()

        # batch_size x seq_length x hidden_size
        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_output)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_output)), dim=-1)

        type_start_output = F.normalize(self.dropout(self.type_start_linear(type_output)), dim=-1)
        type_end_output = F.normalize(self.dropout(self.type_end_linear(type_output)), dim=-1)
        type_span_output = F.normalize(self.dropout(self.type_span_linear(type_output)), dim=-1)

        # batch_size x seq_length x seq_length x hidden_size * 2
        span_hidden = torch.cat(
            [
                token_output.unsqueeze(2).expand(-1, -1, seq_length, -1),
                token_output.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )
        range_vector = torch.arange(seq_length, device=token_output.device)
        span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
        span_width = span_width * (span_width > 0)
        span_width = span_width.clamp(0, self.max_span_length)

        # seq_length x seq_length x hidden_size
        span_width_embeddings = self.width_embedding(span_width)
        concat_span_outputs = torch.cat([
            span_hidden, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3
        )
        # batch_size x seq_length x seq_length x hidden_size
        token_span_output = F.normalize(
            self.dropout(self.token_span_linear(concat_span_outputs)), dim=-1
        )

        start_scores = self.start_logit_scale.exp() * (type_start_output.unsqueeze(0) @ token_start_output.transpose(1, 2))
        end_scores = self.end_logit_scale.exp() * (type_end_output.unsqueeze(0) @ token_end_output.transpose(1, 2))
        span_scores = self.span_logit_scale.exp() * (token_span_output @ type_span_output.T)
        span_scores = span_scores.permute(0, 3, 1, 2)

        if labels is not None:
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
    
    def save_pretrained(self, path: str):
        """Save model, tokenizer configs, and model state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(str(path))
        
        # Save token and type encoder models
        self.token_encoder.save_pretrained(str(path / "token_encoder"))
        self.type_encoder.save_pretrained(str(path / "type_encoder"))
        
        # Save model state dict (span model specific weights)
        torch.save(self.state_dict(), path / "model.pt")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "SpanModel":
        """Load model from saved checkpoint."""
        path = Path(path)
        
        # Load config
        config = SpanModelConfig.from_pretrained(str(path))
        
        # Initialize model with config
        model = cls(config)
        
        # Load token and type encoder models (matching __init__ with add_pooling_layer=False)
        token_config = AutoConfig.from_pretrained(model.config.token_encoder)
        type_config = AutoConfig.from_pretrained(model.config.type_encoder)
        
        if "mt5" in model.config.token_encoder:
            model.token_encoder = MT5EncoderModel.from_pretrained(str(path / "token_encoder"), config=token_config)
        else:
            model.token_encoder = AutoModel.from_pretrained(str(path / "token_encoder"), config=token_config)
        if "mt5" in model.config.type_encoder:
            model.type_encoder = MT5EncoderModel.from_pretrained(str(path / "type_encoder"), config=type_config)
        else:
            model.type_encoder = AutoModel.from_pretrained(str(path / "type_encoder"), config=type_config)
        
        state_dict = torch.load(path / "model.pt", map_location="cpu")
        model_state_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if k in model_state_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
        
        return model

    def gradient_checkpointing_enable(self):
        self.token_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

