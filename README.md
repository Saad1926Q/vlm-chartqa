# VLM-ChartQA

This is the first time I will be using GRPO, and the goal is to get hands-on with it by training a VLM on ChartQA(tho I realized that the model performs pretty decent out of the box also).

The rough idea is to compare three settings - baseline, SFT, and RL - and see how much gains we can squeeze out.

Along the way, hoping to get a better feel for reward function design and the various knobs in GRPO.

Using Qwen3-VL-8B-Instruct (via Unsloth 4-bit quant) fine-tuned with LoRA.

## Plan

1. Eval baseline model to get a reference score
2. Train with SFT, eval again - see how much it improves
3. Take the SFT checkpoint, train with GRPO, eval again 
4. Maybe try CoT SFT + GRPO 

## A bit about the GRPO Training

I guess a little bit of background about the task would be cool. So the task is to take images of charts like bar graphs, pie charts etc and be able to accurately respond to questions based on them.

So, first I trained the model using SFT on the training set of ChartQA.

Then I used this model on which I had run SFT for GRPO on 2000 samples from the dataset which they created in [this paper](https://arxiv.org/pdf/2510.10973) for 1 epoch.

Let's take a look at the GRPO setup:-

So, first of all speaking of the reward functions which is like the bread and butter of this thing. I used 4 reward functions for GRPO. Some of them were inspired by the example in the unsloth notebook for VLM GRPO and some were taken from [this paper](https://arxiv.org/pdf/2510.10973). In the paper they make the model respond in this format - it predicts the type of chart, then it also creates a table in json format which basically tries to capture the information present in a chart and then there is some standard stuff ie the reasoning trace and the final answer itself.

1. Formatting reward function: This reward function basically rewards the model if it responds in the right format. So basically we have four sections ie chart type, table, reasoning and answer. For each section present in the response we add 1 to the reward. There is also a small penalty if the response is mostly junk like repeated `addCriterion` tokens or excessive newlines - this is something the model has been known to do, and the same penalty also shows up in the official unsloth VLM GRPO notebook.

2. Correctness reward function: In this reward function we basically look at the correctness of the final answer. This is done by first of all extracting the final answer from the response and then we use relaxed correctness (which we'll talk about in just a bit) to determine whether the answer is correct or not. We reward a value of 2 if the answer is correct otherwise 0 reward.

3. Chart type reward function: If the model predicts the correct chart type then reward 1 else 0.

4. Table Reward: This reward function is based on the ability of the model to construct the table from the chart. We first try to parse the predicted table as JSON - if that fails, the reward is 0. Otherwise we give a base reward of 0.5 for producing valid JSON, and then add two more quantities on top:
   a. column header accuracy: the fraction of ground truth column names that show up in the predicted columns.
   b. cell accuracy: for each row we compute the fraction of cells that exactly match the ground truth row, and then average this across all rows.

So the table reward maxes out at 2.5 (0.5 for valid JSON + 1 for columns + 1 for cells).

## Evaluation

The evaluation was done on the test set of the ChartQA dataset on hugging face. ChartQA is a benchmark for question answering over charts - given a chart image and a natural language question, the model has to produce the answer. Answers can be numeric (with or without units like %), yes/no, or short text spans.

Regarding the evaluation metric, I use the relaxed correctness metric which is very commonly used for evaluation in the ChartQA task. The idea is pretty simple - for numeric answers we allow up to 5% relative error (so "12.4" is considered correct if the gold is "12.5"), and for non-numeric answers we just fall back to case-insensitive exact match. This way the model isn't unfairly penalized for things like "12.5%" vs "0.125" or minor formatting differences.


## Results

| Stage | Accuracy |
|-------|----------|
| Baseline (Qwen3-VL-8B-Instruct, 4-bit) | 80.1% (2002/2500) |
| Post-SFT | 82.6% (2065/2500) |
| Post-GRPO (on top of SFT) | 82.7% (2067/2500) |


## Setup

```bash
# Clone and install
git clone https://github.com/Saad1926Q/vlm-chartqa && cd vlm-chartqa
uv sync

# Authenticate
hf auth login
wandb login
```

**Run SFT training:**
```bash
uv run python -m vlm_chartqa.train.sft --batch_size <bs> --grad_accum_steps <ga> --push_to_hub --hub_model_id your-hf-username/your-sft-model --use_wandb --wandb_project vlm-sft --wandb_run_name my-sft-run
```

**Run GRPO training:**

Pass `--lora_path` pointing to a checkpoint (local dir or HF repo) so GRPO starts from there rather than the base model.
```bash
uv run python -m vlm_chartqa.train.grpo --lora_path your-hf-username/your-checkpoint --batch_size <bs> --grad_accum_steps <ga> --num_generations <g> --push_to_hub --hub_model_id your-hf-username/your-grpo-model --use_wandb --wandb_project vlm-grpo --wandb_run_name my-grpo-run
```

**Run evaluation:**

Pass `--grpo` only when evaluating a GRPO-trained model (it expects the structured output format). To start from a checkpoint pass `--lora_path` (local dir or HF repo); skip it to evaluate the base model.
```bash
# Baseline (no checkpoint)
uv run python -m vlm_chartqa.eval.eval

# From a checkpoint
uv run python -m vlm_chartqa.eval.eval --lora_path your-hf-username/your-checkpoint

# GRPO checkpoint
uv run python -m vlm_chartqa.eval.eval --grpo --lora_path your-hf-username/your-grpo-checkpoint
```
