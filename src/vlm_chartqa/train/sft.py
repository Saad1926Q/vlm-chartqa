import argparse
import os

from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from vlm_chartqa.config import (
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_GRAD_ACCUM_STEPS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
)
from vlm_chartqa.dataset import prepare_dataset
from vlm_chartqa.model import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="sft_lora")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="vlm-sft")
parser.add_argument("--wandb_run_name", type=str, default=None)
args = parser.parse_args()

if args.push_to_hub:
    if not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is set")

if args.use_wandb:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

model, tokenizer = load_model()

FastVisionModel.for_training(model)

train_dataset = prepare_dataset(mode="sft")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM_STEPS,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LEARNING_RATE,
        warmup_steps=5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=True,
        report_to="wandb" if args.use_wandb else "none",
        output_dir="outputs",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=SFT_MAX_SEQ_LEN,
    ),
)

trainer.train()

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

if args.push_to_hub:
    model.push_to_hub(args.hub_model_id)
    tokenizer.push_to_hub(args.hub_model_id)
