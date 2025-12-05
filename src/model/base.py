import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional
from transformers.file_utils import ModelOutput

@dataclass 
class SpanModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None

def mlp(input_size, output_size, dropout):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(output_size, output_size),
    )

