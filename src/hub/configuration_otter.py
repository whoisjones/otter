"""Configuration for Otter span-based multilingual NER models."""

from transformers import AutoConfig, PretrainedConfig


def _to_config(value):
    """Turn a serialized sub-config dict back into a PretrainedConfig."""
    if value is None or isinstance(value, PretrainedConfig):
        return value
    value = dict(value)
    model_type = value.pop("model_type", None)
    if model_type is None:
        raise ValueError("Encoder sub-config is missing a 'model_type' key.")
    # Keys transformers writes for bookkeeping but AutoConfig.for_model rejects.
    for key in ("transformers_version", "_attn_implementation_autoset"):
        value.pop(key, None)
    return AutoConfig.for_model(model_type, **value)


class OtterConfig(PretrainedConfig):
    """Shared config for the bi-encoder and cross-encoder Otter models.

    The encoder configs are embedded (`token_encoder_config`, `type_encoder_config`)
    so the model is fully self-contained: loading never has to reach out to the base
    encoder repositories, and the checkpoint's own vocabulary size -- which differs
    from the base encoder for the cross-encoder, whose tokenizer gains a `[LABEL]`
    token -- is always the one used.
    """

    model_type = "otter"

    def __init__(
        self,
        token_encoder: str = None,
        type_encoder: str = None,
        token_encoder_config=None,
        type_encoder_config=None,
        architecture: str = "bi_encoder",
        type_encoder_pooling: str = "cls",
        loss_fn: str = "bce",
        max_span_length: int = 30,
        max_seq_length: int = 512,
        linear_hidden_size: int = 384,
        span_width_embedding_size: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 0.03,
        prediction_threshold: float = 0.5,
        start_loss_weight: float = 0.0,
        end_loss_weight: float = 0.0,
        span_loss_weight: float = 1.0,
        bce_start_pos_weight: float = None,
        bce_end_pos_weight: float = None,
        bce_span_pos_weight: float = None,
        focal_alpha: float = None,
        focal_gamma: float = None,
        contrastive_threshold_loss_weight: float = None,
        contrastive_span_loss_weight: float = None,
        contrastive_tau: float = None,
        dice_smooth: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_encoder = token_encoder
        self.type_encoder = type_encoder
        self.token_encoder_config = _to_config(token_encoder_config)
        self.type_encoder_config = _to_config(type_encoder_config)
        self.architecture = architecture
        self.loss_fn = loss_fn
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_span_length = max_span_length
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.linear_hidden_size = linear_hidden_size
        self.span_width_embedding_size = span_width_embedding_size
        self.init_temperature = init_temperature
        self.type_encoder_pooling = type_encoder_pooling
        self.prediction_threshold = prediction_threshold
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.bce_start_pos_weight = bce_start_pos_weight
        self.bce_end_pos_weight = bce_end_pos_weight
        self.bce_span_pos_weight = bce_span_pos_weight
        self.contrastive_threshold_loss_weight = contrastive_threshold_loss_weight
        self.contrastive_span_loss_weight = contrastive_span_loss_weight
        self.contrastive_tau = contrastive_tau
        self.dice_smooth = dice_smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight


class OtterBiEncoderConfig(OtterConfig):
    model_type = "otter-bi-encoder"

    def __init__(self, architecture: str = "bi_encoder", **kwargs):
        super().__init__(architecture=architecture, **kwargs)


class OtterCrossEncoderConfig(OtterConfig):
    model_type = "otter-cross-encoder"

    def __init__(self, architecture: str = "cross_encoder", **kwargs):
        super().__init__(architecture=architecture, **kwargs)


# Kept so checkpoints written by earlier versions of the training code, which
# referenced `configuration_otter.SpanModelConfig`, still resolve.
SpanModelConfig = OtterConfig
