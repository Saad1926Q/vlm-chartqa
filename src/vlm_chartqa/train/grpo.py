from unsloth import FastVisionModel
import argparse
import os

from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from vlm_chartqa.config import (
    GRPO_BATCH_SIZE,
    GRPO_EPOCHS,
    GRPO_GRAD_ACCUM_STEPS,
    GRPO_LEARNING_RATE,
    GRPO_NUM_GENERATIONS,
    GRPO_MAX_PROMPT_LEN,
    GRPO_MAX_COMPLETION_LEN,
)
from vlm_chartqa.dataset import prepare_dataset
from vlm_chartqa.model import load_model
from vlm_chartqa.train.rewards import (
    correctness_reward_func,
    formatting_reward_func,
    chart_type_reward_func,
    table_reward_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="grpo_lora")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=GRPO_BATCH_SIZE)
parser.add_argument("--grad_accum_steps", type=int, default=GRPO_GRAD_ACCUM_STEPS)
parser.add_argument("--num_generations", type=int, default=GRPO_NUM_GENERATIONS)
parser.add_argument("--epochs", type=int, default=GRPO_EPOCHS)
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="vlm-grpo")
parser.add_argument("--wandb_run_name", type=str, default=None)
args = parser.parse_args()

if args.push_to_hub:
    if not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is set")


if args.use_wandb:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

training_args = GRPOConfig(
    learning_rate=GRPO_LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum_steps,
    num_generations=args.num_generations,
    max_prompt_length=GRPO_MAX_PROMPT_LEN,
    max_completion_length=GRPO_MAX_COMPLETION_LEN,
    num_train_epochs=args.epochs,
    save_steps=60,
    max_grad_norm=0.1,
    report_to="wandb" if args.use_wandb else "none",
    output_dir="outputs",
    bf16=True,
    # Below enables GSPO:
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",
)


# Required to avoid unsloth errors
training_args.unsloth_num_chunks = -1
training_args.unsloth_grpo_mini_batch = None
training_args.unsloth_logit_chunk_multiplier = None
training_args.vllm_sampling_params = None

model, tokenizer = load_model(lora_path=args.lora_path)

train_dataset = prepare_dataset()

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    # Pass the processor to handle multimodal inputs
    processing_class=tokenizer,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
        chart_type_reward_func,
        table_reward_fn,
    ],
    train_dataset=train_dataset,
)

trainer.train()

# Save
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Push to hub
if args.push_to_hub:
    model.push_to_hub(args.hub_model_id)
    tokenizer.push_to_hub(args.hub_model_id)
