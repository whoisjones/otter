from .collator import (
    EvalCollatorBiEncoder,
    EvalCollatorContrastiveBiEncoder,
    EvalCollatorContrastiveCrossEncoder,
    EvalCollatorCrossEncoder,
    TrainCollatorBiEncoder,
    TrainCollatorContrastiveBiEncoder,
    TrainCollatorContrastiveCrossEncoder,
    TrainCollatorCrossEncoder,
)
from .config import ARCHITECTURES, SpanModelConfig, is_bi_encoder, is_contrastive
from .logger import setup_logger
from .metrics import add_batch_metrics, compute_span_predictions, finalize_metrics
from .model import (
    OtterBiEncoderModel,
    OtterContrastiveBiEncoderModel,
    OtterContrastiveCrossEncoderModel,
    OtterCrossEncoderModel,
)
from .trainer import evaluate, train

__all__ = [
    "OtterBiEncoderModel",
    "OtterCrossEncoderModel",
    "OtterContrastiveBiEncoderModel",
    "OtterContrastiveCrossEncoderModel",
    "ARCHITECTURES",
    "SpanModelConfig",
    "is_bi_encoder",
    "is_contrastive",
    "TrainCollatorBiEncoder",
    "EvalCollatorBiEncoder",
    "TrainCollatorCrossEncoder",
    "EvalCollatorCrossEncoder",
    "TrainCollatorContrastiveBiEncoder",
    "EvalCollatorContrastiveBiEncoder",
    "TrainCollatorContrastiveCrossEncoder",
    "EvalCollatorContrastiveCrossEncoder",
    "compute_span_predictions",
    "add_batch_metrics",
    "finalize_metrics",
    "train",
    "evaluate",
    "setup_logger",
]
