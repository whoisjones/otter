from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    """
    Arguments for Binder.
    """

    token_encoder: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    type_encoder: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate for hidden states."}
    )
    linear_hidden_size: int = field(
        default=128, metadata={"help": "Size of the last linear layer."}
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
    type_encoder_pooling: str = field(
        default="cls", metadata={"help": "Pooling method for type encoder. Options: 'cls' (uses CLS token) or 'mean' (mean pooling)."}
    )
    prediction_threshold: float = field(
        default=0.4,
        metadata={
            "help": "Threshold for span predictions (lower = higher recall, higher = higher precision). Default 0.4."
        },
    )
    use_pos_weight: bool = field(
        default=True,
        metadata={
            "help": "Whether to use positive class weight in loss. Default True."
        },
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
                extension = self.train_file.split(".")[-1]
                assert extension == "jsonl", "`train_file` should be a jsonl file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "jsonl", "`validation_file` should be a jsonl file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension == "jsonl", "`test_file` should be a jsonl file."