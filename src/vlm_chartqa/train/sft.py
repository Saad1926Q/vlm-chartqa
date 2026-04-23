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

training_args = parser.add_argument_group("training")
training_args.add_argument("--batch_size", type=int, default=SFT_BATCH_SIZE)
training_args.add_argument("--grad_accum_steps", type=int, default=SFT_GRAD_ACCUM_STEPS)
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
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
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
