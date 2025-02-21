import logging
import os
from pathlib import Path

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .finetune.arguments import (
    ModelArguments,
    DataArguments,
    MyTrainingArguments as TrainingArguments,
)
from .finetune.data import MyTrainDataset, CollatorFunction
from .finetune.modeling import MyModel
from .finetune.trainer import MyTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        # Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # load_tokenizer_and_model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = MyModel(
        model_name=model_args.model_name_or_path,
    )
    # from transformers import AutoConfig
    # num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    # logger.info("Config: %s", config)

    # load_train_dataset
    train_dataset = MyTrainDataset(args=data_args, tokenizer=tokenizer)

    # load_data_collator
    data_collator = CollatorFunction(
        tokenizer,
        max_len=data_args.max_len,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()

    # Re-save the tokenizer to the same directory, so we can share model easily on huggingface.co/models
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
