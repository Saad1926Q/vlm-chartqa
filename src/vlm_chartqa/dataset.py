from datasets import load_dataset
from tqdm import tqdm

from vlm_chartqa.config import (
    DATASET,
    DATASET_GRPO,
    DATASET_SIZE,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    CHART_TYPE_START,
    CHART_TYPE_END,
    TABLE_START,
    TABLE_END,
)


def _prepare_image(example):
    image = example["image"].resize((512, 512))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _process_sft(example):
    image = _prepare_image(example)

    text = f"{example['query'].strip()} "

    system_prompt = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["label"][0]}],
        },
    ]

    return {
        "messages": prompt,
        "image": image,
        "answer": example["label"][0],
    }


def _process_eval(example):
    image = _prepare_image(example)

    system_prompt = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["query"].strip()},
            ],
        },
    ]

    return {
        "prompt": prompt,
        "image": image,
        "answer": example["label"][0],
    }


def _process_grpo(example):
    image = _prepare_image(example)

    text = (
        f"You are a vision-language assistant. You are given a chart image and a query about the chart. "
        f"Think step-by-step about how to answer the query based on the chart, then provide the final answer.\n\n"
        f"Respond with exactly four blocks in this order and nothing else:\n\n"
        f"{CHART_TYPE_START}\n"
        f"One word from: line, bar, stacked bar, pie, histogram, scatterplot, area, stacked area, bubble, treemap.\n"
        f"{CHART_TYPE_END}\n"
        f"{TABLE_START}\n"
        f"Only a JSON object with this exact schema and nothing else inside the tags:\n"
        f'{{"columns": [...], "rows": [[...], [...], ..., [...]]}}\n'
        f"\"columns\" is a list of column headers. \"rows\" is a list-of-lists, one inner list per data row. "
        f"No HTML, Markdown, or commentary inside this block.\n"
        f"{TABLE_END}\n"
        f"{REASONING_START}\n"
        f"Reason step-by-step:\n"
        f"<step-1>: Briefly describe what the chart shows.\n"
        f"<step-2>: Gather the values from the chart needed to answer the query.\n"
        f"<step-3>: Break the query into smaller parts and verify each against the data.\n"
        f"...\n"
        f"<step-n>: Do the final calculation or reasoning to derive the answer.\n"
        f"{REASONING_END}\n"
        f"{SOLUTION_START}\n"
        f"Final answer on a single line.\n"
        f"{SOLUTION_END}\n\n"
        f"Query: {example['query'].strip()}"
    )

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]

    return {
        "prompt": prompt,
        "image": image,
    }


def prepare_dataset(mode="grpo", split=None):
    if split is None:
        if mode == "sft":
            split = "train"
        else:
            split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
    dataset = load_dataset(DATASET_GRPO if mode == "grpo" else DATASET, split=split)
    if mode == "grpo":
        dataset = dataset.map(_process_grpo)
        cols = ["prompt", "image", "label", "chart_type", "table"]
        return dataset.select_columns(cols)
    elif mode == "sft":
        return [_process_sft(ex) for ex in tqdm(dataset, desc="Processing SFT")]
    else:  # eval
        dataset = dataset.map(_process_eval)
        cols = ["prompt", "image", "answer"]
        return dataset.select_columns(cols)
