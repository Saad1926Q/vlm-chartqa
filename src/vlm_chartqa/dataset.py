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
        f"{example['query'].strip()} "
        f"First identify the chart type between {CHART_TYPE_START} and {CHART_TYPE_END}, "
        f"then reconstruct the chart data as a JSON table between {TABLE_START} and {TABLE_END} "
        f"with keys 'columns' (list of column headers) and 'rows' (list of rows, each row is a list of values), "
        f"then provide your reasoning between {REASONING_START} and {REASONING_END}, "
        f"and finally your answer between {SOLUTION_START} and {SOLUTION_END}."
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
