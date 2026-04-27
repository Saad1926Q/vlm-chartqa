# VLM-ChartQA

My first attempt at using GRPO to train a VLM on ChartQA. Plan is to run mini experiments, learn from mistakes, and document everything in one place.

The rough idea is to compare three settings - baseline, SFT, and RL - and see how much gains we can squeeze out.

Along the way, hoping to get a better feel for reward function design and the various knobs in GRPO.

Using Qwen3-VL-8B-Instruct (via Unsloth 4-bit quant) fine-tuned with LoRA.

## Setup

```bash
# Clone and install
git clone https://github.com/Saad1926Q/vlm-chartqa && cd vlm-chartqa
uv sync

# Authenticate
huggingface-cli login
wandb login
```

**Run GRPO training:**
```bash
uv run python -m vlm_chartqa.train.grpo --batch_size 4 --grad_accum_steps 8 --num_generations 4 --push_to_hub --hub_model_id your-hf-username/your-model-name --use_wandb --wandb_project vlm-grpo --wandb_run_name my-run
```


## Plan

1. Eval baseline model to get a reference score
2. Train with SFT, eval again - see how much it improves
3. Take the SFT checkpoint, train with GRPO, eval again 
4. If compute allows, try CoT SFT + GRPO - generate reasoning traces with a stronger model, SFT on those, then GRPO on top

## Results

| Stage | Accuracy |
|-------|----------|
| Baseline (Qwen3-VL-8B-Instruct, 4-bit) | 80.1% (2002/2500) |
| Post-SFT | 82.6% (2065/2500) |
| Post-GRPO | TBD |

## Learnings

So initially I went with exact match as the eval metric and then realized it's actually not a great way to evaluate the model. Like a model answering "12.5%" vs "0.125" shouldn't really be penalized. After looking up how ChartQA is evaluated everywhere, I came across the relaxed correctness metric. The idea is that for numeric answers you allow up to 5% relative error, and for non-numeric you just fall back to exact match (case-insensitive). Btw one problem I noticed with the implementation is that it should have been `prediction_float is not None and target_float is not None` but instead it's just `and target_float`. But since this is the implementation used everywhere, I kept it for uniformity.
