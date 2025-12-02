from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments for Binder.
    """

    token_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Required if model_checkpoint is not provided."}
    )
    type_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Required if model_checkpoint is not provided."}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate for hidden states."}
    )
    linear_hidden_size: int = field(
        default=128, metadata={"help": "Size of the last linear layer."}
    )
    span_width_embedding_size: int = field(
        default=128, metadata={"help": "Size of the span width embedding."}
    )
    init_temperature: float = field(
        default=0.03, metadata={"help": "Initial temperature for the logits."}
    )
    start_loss_weight: float = field(
        default=0.2, metadata={"help": "Weight for the start loss."}
    )
    end_loss_weight: float = field(
        default=0.2, metadata={"help": "Weight for the end loss."}
    )
    span_loss_weight: float = field(
        default=0.6, metadata={"help": "Weight for the span loss."}
    )
    start_pos_weight: Optional[float] = field(
        default=None, metadata={"help": "Positive weight for the start loss. If None, no pos_weight is applied."}
    )
    end_pos_weight: Optional[float] = field(
        default=None, metadata={"help": "Positive weight for the end loss. If None, no pos_weight is applied."}
    )
    span_pos_weight: Optional[float] = field(
        default=None, metadata={"help": "Positive weight for the span loss. If None, no pos_weight is applied."}
    )
    type_encoder_pooling: str = field(
        default="cls", metadata={"help": "Pooling method for type encoder. Options: 'cls' (uses CLS token) or 'mean' (mean pooling)."}
    )
    prediction_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Threshold for span predictions (lower = higher recall, higher = higher precision). Default 0.4."
        },
    )
    loss_fn: str = field(
        default="bce",
        metadata={
            "help": "The loss function to use. Can be 'bce' or 'focal'."
        },
    )
    focal_alpha: float = field(
        default=0.75,
        metadata={
            "help": "Alpha for the focal loss."
        },
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Gamma for the focal loss."
        },
    )
    model_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pretrained span model checkpoint to load from. If provided, the model will be loaded from this checkpoint instead of being initialized from scratch."
        },
    )

    def __post_init__(self):
        """Validate that either model_checkpoint is provided, or both token_encoder and type_encoder are provided."""
        if self.model_checkpoint is None:
            if self.token_encoder is None or self.type_encoder is None:
                raise ValueError(
                    "Either 'model_checkpoint' must be provided, or both 'token_encoder' and 'type_encoder' must be provided."
                )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use, from which it will decide entity types to use."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    preprocessing_num_workers: Optional[int] = field(
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
        metadata={
            "help": "The maximum length of an entity span."
        },
    )
    annotation_format: str = field(
        default='text',
        metadata={
            "help": "The format of the annotation. Can be 'text' or 'tokens'."
        },
    )
    loss_masking: str = field(
        default='none',
        metadata={
            "help": "The method to mask the loss. Can be 'none', 'all_spans' or 'subwords'."
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
    """
    Extended TrainingArguments with custom fields for this project.
    """
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Number of evaluation steps to wait before early stopping if no improvement."}
    )