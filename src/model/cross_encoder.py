import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig, PreTrainedModel, MT5EncoderModel, AutoTokenizer
from pathlib import Path

from .base import SpanModelOutput, mlp
from ..config import SpanModelConfig
from ..loss import BCELoss, FocalLoss, DiceLoss, DiceFocalLoss

class OtterCrossEncoderModel(PreTrainedModel):
    """Dual encoder with span marker module."""

    config_class = SpanModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        token_config = AutoConfig.from_pretrained(self.config.token_encoder)

        self.max_span_length = config.max_span_length
        self.dropout = nn.Dropout(config.dropout)
        self.linear_hidden_size = config.linear_hidden_size
        self.config.pruned_heads = getattr(token_config, "pruned_heads", {})

        self.type_linear = mlp(token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_start_linear = mlp(token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_end_linear = mlp(token_config.hidden_size, config.linear_hidden_size, config.dropout)
        self.token_span_linear = mlp(config.linear_hidden_size * 2 + config.span_width_embedding_size, config.linear_hidden_size, config.dropout)
        self.width_embedding = nn.Embedding(config.max_span_length + 1, config.span_width_embedding_size, padding_idx=0)
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        if "mt5" in config.token_encoder:
            self.token_encoder = MT5EncoderModel(token_config)
        else:
            self.token_encoder = AutoModel.from_config(token_config)
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
        labels: dict = None,
        **kwargs
    ):
        encoder_outputs = self.token_encoder(**token_encoder_inputs).last_hidden_state
        token_hidden = encoder_outputs[:, labels["text_start_index"]:, :]
        type_hidden = encoder_outputs[:, labels["label_token_subword_positions"], :]

        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_hidden)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_hidden)), dim=-1)
        type_output = F.normalize(self.dropout(self.type_linear(type_hidden)), dim=-1)

        start_scores = self.start_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_start_output, type_output)
        end_scores = self.end_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_end_output, type_output)
        
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
        span_scores = self.span_logit_scale.exp() * torch.einsum("BSH,BCH->BCS", token_span_output, type_output)

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
    
    def save_pretrained(self, path: str, **kwargs):
        path = Path(path)
        tokenizer = kwargs.pop("tokenizer", None)
        super().save_pretrained(str(path), **kwargs)
        self.token_encoder.config.to_json_file(str(path / "token_encoder_config.json"))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(path))

    @classmethod
    def from_pretrained(cls, path: str, *model_args, config=None, **kwargs):
        path = Path(path)

        span_cfg = SpanModelConfig.from_pretrained(str(path))
        model = cls(span_cfg)

        weights_file = path / "model.safetensors"
        if not weights_file.exists():
            weights_file = path / "model.bin"

        # Load encoder weights from HuggingFace to get proper pretrained weights
        if "mt5" in span_cfg.token_encoder:
            model.token_encoder = MT5EncoderModel.from_pretrained(span_cfg.token_encoder, config=model.token_encoder.config)
        else:
            model.token_encoder = AutoModel.from_pretrained(span_cfg.token_encoder, config=model.token_encoder.config)

        if weights_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_file))
        else:
            state_dict = torch.load(str(weights_file), map_location="cpu")

        # Resize token embeddings to match checkpoint vocab size
        current_vocab_size = model.token_encoder.get_input_embeddings().weight.shape[0]
        for key, val in state_dict.items():
            if "token_encoder" in key and val.ndim == 2 and any(
                tok in key for tok in ("word_embeddings.weight", "tok_embeddings.weight", "shared.weight", "embed_tokens.weight")
            ):
                if val.shape[0] != current_vocab_size:
                    model.token_encoder.resize_token_embeddings(val.shape[0])
                break

        model.load_state_dict(state_dict, strict=False)
        return model

    def gradient_checkpointing_enable(self):
        self.token_encoder.gradient_checkpointing_enable()

