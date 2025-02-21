import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

"""
    Using [`HfArgumentParser`] we can turn this class into arguments that can be specified on the command line.
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) 
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    # Other Model related Arguments
    # 其他模型相关的参数


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to train data"})
    max_example_num_per_dataset: int = field(
        default=None, metadata={"help": "the max number of examples for each dataset"}
    )
    # Other Data related Arguments
    # 其他 数据相关的参数

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(
                f"cannot find file: {self.train_data}, please set a true path"
            )


@dataclass
class MyTrainingArguments(TrainingArguments):

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    # Other Training related Arguments
    # 其他 Training 相关的参数
