from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    architecture: str | None = field(
        default=None,
        metadata={
            "help": "Model architecture. One of: 'bi_encoder', 'cross_encoder', 'contrastive_bi_encoder', 'contrastive_cross_encoder'."
        },
    )
    token_encoder: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Required if model_checkpoint is not provided."
        },
    )
    type_encoder: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Required if model_checkpoint is not provided."
        },
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for hidden states."})
    linear_hidden_size: int = field(
        default=128, metadata={"help": "Size of the last linear layer."}
    )
    span_width_embedding_size: int = field(
        default=128, metadata={"help": "Size of the span width embedding."}
    )
    init_temperature: float = field(
        default=0.03, metadata={"help": "Initial temperature for the logits."}
    )
    start_loss_weight: float = field(default=0.2, metadata={"help": "Weight for the start loss."})
    end_loss_weight: float = field(default=0.2, metadata={"help": "Weight for the end loss."})
    span_loss_weight: float = field(default=0.6, metadata={"help": "Weight for the span loss."})
    bce_start_pos_weight: float | None = field(
        default=None,
        metadata={"help": "Positive weight for the start loss. If None, no pos_weight is applied."},
    )
    bce_end_pos_weight: float | None = field(
        default=None,
        metadata={"help": "Positive weight for the end loss. If None, no pos_weight is applied."},
    )
    bce_span_pos_weight: float | None = field(
        default=None,
        metadata={"help": "Positive weight for the span loss. If None, no pos_weight is applied."},
    )
    contrastive_threshold_loss_weight: float = field(
        default=0.5, metadata={"help": "Weight for the threshold loss."}
    )
    contrastive_span_loss_weight: float = field(
        default=0.5, metadata={"help": "Weight for the span loss."}
    )
    contrastive_tau: float = field(
        default=1.0, metadata={"help": "Temperature for the contrastive loss."}
    )
    type_encoder_pooling: str = field(
        default="cls",
        metadata={
            "help": "Pooling method for type encoder. Options: 'cls' (uses CLS token) or 'mean' (mean pooling)."
        },
    )
    prediction_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Threshold for span predictions (lower = higher recall, higher = higher precision). Default 0.4."
        },
    )
    loss_fn: str = field(
        default="bce",
        metadata={"help": "The loss function to use. One of 'bce', 'focal', 'dice', 'dice_focal'."},
    )
    focal_alpha: float = field(
        default=0.75,
        metadata={"help": "Alpha for the focal loss."},
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={"help": "Gamma for the focal loss."},
    )
    dice_smooth: float = field(
        default=1.0,
        metadata={"help": "Smoothing constant for Dice loss."},
    )
    dice_weight: float = field(
        default=0.5,
        metadata={"help": "Weight of the Dice component in DiceFocal loss."},
    )
    focal_weight: float = field(
        default=0.5,
        metadata={"help": "Weight of the Focal component in DiceFocal loss."},
    )
    contrastive_threshold_token: str = field(
        default="label_token",
        metadata={
            "help": "Token type used as prediction threshold in contrastive cross-encoder. One of 'label_token' or 'cls'."
        },
    )
    model_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "Path to a pretrained span model checkpoint to load from. If provided, the model will be loaded from this checkpoint instead of being initialized from scratch."
        },
    )

    def __post_init__(self):
        if self.model_checkpoint is None and (
            self.token_encoder is None or self.type_encoder is None
        ):
            raise ValueError(
                "Either 'model_checkpoint' must be provided, or both 'token_encoder' and "
                "'type_encoder' must be provided."
            )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use, from which it will decide entity types to use."
        }
    )
    train_file: str | None = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: str | None = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    test_file: str | None = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the perplexity on (a text file)."
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_span_length: int = field(
        default=30,
        metadata={"help": "The maximum length of an entity span."},
    )
    annotation_format: str = field(
        default="text",
        metadata={"help": "The format of the annotation. Can be 'text' or 'tokens'."},
    )
    loss_masking: str = field(
        default="none",
        metadata={"help": "The method to mask the loss. Can be 'none', 'all_spans' or 'subwords'."},
    )
    language_weights: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": "Sampling multipliers per language, keyed by the language prefix of the "
            'example id, e.g. {"tha": 10, "cmn": 3}. A language holding 2% of the '
            "sentences at weight 5 ends up near 10% of the draws; languages left out keep "
            "weight 1. None samples the corpus uniformly."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                if isinstance(self.train_file, list):
                    for file in self.train_file:
                        extension = file.split(".")[-1]
                        assert extension == "jsonl", "`train_file` should be a jsonl file."
                elif isinstance(self.train_file, str):
                    extension = self.train_file.split(".")[-1]
                    assert extension == "jsonl", "`train_file` should be a jsonl file."
            if self.validation_file is not None:
                if isinstance(self.validation_file, list):
                    for file in self.validation_file:
                        extension = file.split(".")[-1]
                        assert extension == "jsonl", "`validation_file` should be a jsonl file."
                elif isinstance(self.validation_file, str):
                    extension = self.validation_file.split(".")[-1]
                    assert extension == "jsonl", "`validation_file` should be a jsonl file."
            if self.test_file is not None:
                if isinstance(self.test_file, list):
                    for file in self.test_file:
                        extension = file.split(".")[-1]
                        assert extension == "jsonl", "`test_file` should be a jsonl file."
                elif isinstance(self.test_file, str):
                    extension = self.test_file.split(".")[-1]
                    assert extension == "jsonl", "`test_file` should be a jsonl file."


@dataclass
class CustomTrainingArguments(TrainingArguments):
    early_stopping_patience: int = field(
        default=5,
        metadata={
            "help": "Number of evaluation steps to wait before early stopping if no improvement."
        },
    )
    type_encoder_learning_rate: float | None = field(
        default=None,
        metadata={
            "help": "Learning rate for the type encoder. If None, uses the same learning rate as other parameters."
        },
    )
    linear_layers_learning_rate: float | None = field(
        default=None,
        metadata={
            "help": "Learning rate for linear layers and other non-encoder parameters. If None, uses the same learning rate as other parameters."
        },
    )
