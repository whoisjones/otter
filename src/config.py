import json
import argparse
from pathlib import Path
from typing import Dict, Any

from transformers import PretrainedConfig

class SpanModelConfig(PretrainedConfig):

    def __init__(
        self,
        token_encoder: str = None,
        type_encoder: str = None,
        max_span_length: int = 30,
        linear_hidden_size: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 0.07,
        type_encoder_pooling: str = "cls",
        use_pos_weight: bool = True,
        prediction_threshold: float = 0.4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_encoder = token_encoder
        self.type_encoder = type_encoder
        self.max_span_length = max_span_length
        self.dropout = dropout
        self.linear_hidden_size = linear_hidden_size
        self.init_temperature = init_temperature
        self.type_encoder_pooling = type_encoder_pooling
        self.use_pos_weight = use_pos_weight
        self.prediction_threshold = prediction_threshold