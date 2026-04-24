from datasets import load_dataset

from vlm_chartqa.config import (
    DATASET,
    DATASET_SIZE,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
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
                {"type": "image", "image": image},
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
        f"Provide reasoning between {REASONING_START} and {REASONING_END}, "
        f"then your answer between {SOLUTION_START} and {SOLUTION_END}."
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
        "answer": example["label"][0],
    }


def prepare_dataset(mode="grpo", split=None):
    if split is None:
        if mode == "sft":
            split = "train"
        else:
            split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
    dataset = load_dataset(DATASET, split=split)
    if mode == "grpo":
        dataset = dataset.map(_process_grpo)
        cols = ["prompt", "image", "answer"]
        return dataset.select_columns(cols)
    elif mode == "sft":
        dataset = dataset.map(_process_sft)
        cols = ["messages", "image", "answer"]
        return dataset.select_columns(cols)
    else:  # eval
        dataset = dataset.map(_process_eval)
        cols = ["prompt", "image", "answer"]
        return dataset.select_columns(cols)
