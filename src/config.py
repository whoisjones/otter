from transformers import PretrainedConfig

ARCHITECTURES = (
    "bi_encoder",
    "cross_encoder",
    "contrastive_bi_encoder",
    "contrastive_cross_encoder",
)


def is_bi_encoder(architecture: str) -> bool:
    return "bi_encoder" in architecture


def is_contrastive(architecture: str) -> bool:
    return "contrastive" in architecture


class SpanModelConfig(PretrainedConfig):
    def __init__(
        self,
        token_encoder: str = None,
        type_encoder: str = None,
        type_encoder_pooling: str = "cls",
        loss_fn: str = "bce",
        max_span_length: int = 30,
        linear_hidden_size: int = 384,
        span_width_embedding_size: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 0.03,
        prediction_threshold: float = 0.5,
        start_loss_weight: float = 0.0,
        end_loss_weight: float = 0.0,
        span_loss_weight: float = 1.0,
        bce_start_pos_weight: float = 0.0,
        bce_end_pos_weight: float = 0.0,
        bce_span_pos_weight: float = 0.0,
        focal_alpha: float = 0.0,
        focal_gamma: float = 0.0,
        contrastive_threshold_loss_weight: float = 0.0,
        contrastive_span_loss_weight: float = 0.0,
        contrastive_tau: float = 1.0,
        dice_smooth: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_encoder = token_encoder
        self.type_encoder = type_encoder
        self.loss_fn = loss_fn
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_span_length = max_span_length
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
