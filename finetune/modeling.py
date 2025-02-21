import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class MyModel(nn.Module):

    def __init__(
        self,
        model_name: str = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Distributed or Run in a single GPU
        if not dist.is_initialized():
            raise ValueError(
                "Distributed training has not been initialized for representation all gather."
            )

        self.process_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        reps: Tensor,
    ):
        """input tensor and forward to compute loss"""
        if self.training:
            scores = self.training_compute()
            target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)

        else:
            scores = self.eval_compute()
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def training_compute(self):
        pass

    def eval_compute(self):
        pass

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)
