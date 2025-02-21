import os.path
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments


class MyTrainDataset(Dataset):
    """dataset process"""

    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.train_data, file),
                    split="train",
                )

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset(
                "json", data_files=args.train_data, split="train"
            )

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        return self.dataset[item]


@dataclass
class CollatorFunction(DataCollatorWithPadding):
    """
    Pass batch separately to the actual collator.
    根据自己的数据格式处理
    """

    max_len: int = 128

    def __call__(self, data):
        features = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {"features": features}


@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
        self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
