import json
import re
from typing import Any

from vlm_chartqa.config import (
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    CHART_TYPE_START,
    CHART_TYPE_END,
    TABLE_START,
    TABLE_END,
)
from vlm_chartqa.eval.utils import relaxed_correctness


def formatting_reward_func(completions:list[Any], **kwargs) -> list[float]:
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    chart_type_pattern=f"{CHART_TYPE_START}(.*?){CHART_TYPE_END}"
    table_pattern=f"{TABLE_START}(.*?){TABLE_END}"

    scores = []

    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""

        score = 0.0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        chart_type_matches= re.findall(chart_type_pattern, completion, re.DOTALL)
        table_matches=re.findall(table_pattern, completion, re.DOTALL)

        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0
        if len(chart_type_matches)==1:
            score += 1.0
        if len(table_matches)==1:
            score+=1.0


        # Penalize on excessive addCriterion and newlines
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0
        scores.append(score)
    return scores


def correctness_reward_func(completions: list[Any], label: list[str], **kwargs) -> list[float]:
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    completions = [
        (c[0]["content"] if c else "") if isinstance(c, list) else c
        for c in completions
    ]

    responses = [
        re.findall(answer_pattern, completion, re.DOTALL) for completion in completions
    ]

    return [
        2.0 if len(r) == 1 and relaxed_correctness(r[0].strip(), a.strip()) else 0.0
        for r, a in zip(responses, label)
    ]


def chart_type_reward_func(completions: list[Any], chart_type: list[str], **kwargs) -> list[float]:
    chart_type_pattern = f"{CHART_TYPE_START}(.*?){CHART_TYPE_END}"
    completions = [
        (c[0]["content"] if c else "") if isinstance(c, list) else c
        for c in completions
    ]

    scores = []
    for completion, gt in zip(completions, chart_type):
        matches = re.findall(chart_type_pattern, completion, re.DOTALL)
        if len(matches) == 1 and matches[0].strip().lower() == gt.strip().lower():
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def table_reward_fn(completions:list[Any],table:list[dict[str,Any]],**kwargs):

    def _column_header_accuracy(pred_table:dict[str,Any],gt_table:dict[str,Any]):
        result=0

        n_cols=len(gt_table["columns"])
        pred_cols=set(pred_table["columns"])

        for col in gt_table["columns"]:
            if col in pred_cols:
                result+=1

        result/=n_cols

        return result


    def _cell_accuracy(pred_table:dict[str,Any],gt_table:dict[str,Any]):
        gt_rows=gt_table["rows"]
        pred_rows=pred_table["rows"]
        n_rows=len(gt_rows)

        if n_rows==0:
            return 0.0

        total=0.0
        for gt_row,pred_row in zip(gt_rows,pred_rows):
            if not gt_row:
                continue
            matches=sum(1 for gt_cell,pred_cell in zip(gt_row,pred_row) if gt_cell==pred_cell)
            total+=matches/len(gt_row)

        return total/n_rows


    table_pattern=f"{TABLE_START}(.*?){TABLE_END}"

    completions = [
        (c[0]["content"] if c else "") if isinstance(c, list) else c
        for c in completions
    ]

    scores = []
    for completion, gt in zip(completions, table):
        matches = re.findall(table_pattern, completion, re.DOTALL)

        if len(matches) != 1:
            scores.append(0.0)
            continue

        try:
            pred_table = json.loads(matches[0].strip())
        except (json.JSONDecodeError, ValueError):
            scores.append(0.0)
            continue

        score = 0.5
        score += _column_header_accuracy(pred_table, gt)
        score += _cell_accuracy(pred_table, gt)
        scores.append(score)

    return scores
