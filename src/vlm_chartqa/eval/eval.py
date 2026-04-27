from unsloth import FastVisionModel
import argparse
import re

from tqdm import tqdm
from vlm_chartqa.dataset import prepare_dataset

from vlm_chartqa.config import MAX_SEQ_LEN, MODEL_NAME, SOLUTION_START, SOLUTION_END
from vlm_chartqa.eval.utils import relaxed_correctness

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--mode", type=str, choices=["sft", "grpo"], default="sft")
args = parser.parse_args()

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=args.lora_path if args.lora_path else MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

split = f"test[:{args.num_samples}]" if args.num_samples else "test"
dataset = prepare_dataset(mode="eval", split=split, eval_mode=args.mode)

max_new_tokens = 1024 if args.mode == "grpo" else 64
solution_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

correct = 0
total = 0

for batch in tqdm(
    dataset.iter(batch_size=args.batch_size), total=len(dataset) // args.batch_size
):
    images = batch["image"]
    prompts = batch["prompt"]
    answers = batch["answer"]

    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    inputs = tokenizer(
        images,
        texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=1.0,
        min_p=0.1,
    )

    responses = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    for response, answer in zip(responses, answers):
        if args.mode == "grpo":
            matches = re.findall(solution_pattern, response, re.DOTALL)
            pred = matches[0].strip() if len(matches) == 1 else ""
        else:
            pred = response.strip()
        if relaxed_correctness(pred, answer.strip()):
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.1f}%")
