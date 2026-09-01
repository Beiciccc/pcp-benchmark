"""
Shared utilities for PCP evaluation.

This module keeps parsing, task shuffling, baselines, and metrics identical
across local GPU runs and closed-source API upper-bound runs.
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


CHOICE_RE = re.compile(r"\b([A-Z])\b")


def load_compact_tasks(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tasks" in data:
        return data["tasks"]
    return data


def load_probes(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["probes"]


def split_multiple_choice_prompt(prompt: str) -> str:
    """Return the prompt prefix before answer choices."""
    match = re.search(r"\n\nA[.)]\s", prompt)
    if match:
        return prompt[: match.start()].rstrip()
    return prompt.rstrip()


def format_choices(choices: Sequence[str], labels: Sequence[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def answer_instruction(labels: Sequence[str], prompt_style: str) -> str:
    joined = ", ".join(labels)
    if prompt_style == "json":
        return (
            f'Respond with JSON only, in this exact form: {{"answer":"{labels[0]}"}}. '
            f"The answer must be one of: {joined}."
        )
    return f"Answer with only one letter: {joined}."


def rebuild_choice_prompt(
    original_prompt: str,
    choices: Sequence[str],
    labels: Sequence[str],
    prompt_style: str = "letter",
) -> str:
    prefix = split_multiple_choice_prompt(original_prompt)
    return (
        f"{prefix}\n\n"
        f"{format_choices(choices, labels)}\n\n"
        f"{answer_instruction(labels, prompt_style)}"
    )


def make_shuffled_choice_tasks(
    tasks: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    prompt_styles: Sequence[str] = ("letter",),
    labels: Sequence[str] = ("A", "B", "C", "D"),
) -> List[Dict[str, Any]]:
    """Create deterministic option-order variants for multiple-choice tasks."""
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        for prompt_style in prompt_styles:
            rng = random.Random(seed)
            for task in tasks:
                choices = list(task["choices"])
                indexed = list(enumerate(choices))
                rng.shuffle(indexed)
                shuffled_choices = [choice for _, choice in indexed]
                shuffled_metadata = None
                if "choice_metadata" in task:
                    metadata = list(task["choice_metadata"])
                    shuffled_metadata = [metadata[old_idx] for old_idx, _ in indexed]
                old_to_new = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(indexed)}
                correct_idx = old_to_new[task["correct_idx"]]
                new_task = dict(task)
                new_task["choices"] = shuffled_choices
                if shuffled_metadata is not None:
                    new_task["choice_metadata"] = shuffled_metadata
                new_task["correct_idx"] = correct_idx
                new_task["correct_letter"] = labels[correct_idx]
                new_task["prompt"] = rebuild_choice_prompt(
                    task["prompt"], shuffled_choices, labels, prompt_style
                )
                new_task["shuffle_seed"] = seed
                new_task["prompt_style"] = prompt_style
                new_task["eval_id"] = f"{task['probe_id']}__seed{seed}__{prompt_style}"
                out.append(new_task)
    return out


def parse_choice_answer(response: str, labels: Sequence[str] = ("A", "B", "C", "D")) -> Optional[str]:
    """Parse a model answer without matching letters inside ordinary words.

    The old evaluator used `if "A" in response`, which turns "The answer is C"
    into A. This parser only accepts a standalone label, common answer phrases,
    or JSON-like answer fields.
    """
    if response is None:
        return None

    allowed = "".join(re.escape(label) for label in labels)
    text = unicodedata.normalize("NFKC", response).strip()
    if not text:
        return None
    upper = text.upper()

    patterns = [
        rf'"ANSWER"\s*:\s*"([{allowed}])"',
        rf"'ANSWER'\s*:\s*'([{allowed}])'",
        rf"\bANSWER\b[^\n\r]*?(?:IS|:|=)\s*\**([{allowed}])\**\b",
        rf"\b(?:PHENOMENON|PRAGMATIC PHENOMENON|FEATURE|CATEGORY|LABEL)\b[^\n\r]*?(?:IS|:|=)\s*\**([{allowed}])\**\b",
        rf"\b(?:SALIENT|BEST DESCRIBED AS|BEST ANSWER)\b[^\n\r]*?(?:IS|:|=)\s*\**([{allowed}])\**\b",
        rf"\b(?:CHOOSE|PICK|SELECT)\s*\**([{allowed}])\**\b",
        rf"\bANSWER\s*(?:IS|:|=)?\s*([{allowed}])\b",
        rf"\bOPTION\s*(?:IS|:|=)?\s*([{allowed}])\b",
        rf"\bCHOICE\s*(?:IS|:|=)?\s*([{allowed}])\b",
        rf"^\s*([{allowed}])\s*[\).:\-]?\s*$",
        rf"^\s*#+\s*\**([{allowed}])\**\b",
        rf"(?:^|[\n\r])\s*#+\s*[A-Z][A-Z ]{{0,40}}:\s*\**([{allowed}])\**\b",
        rf"(?:^|[\n\r])\s*#*\s*\**([{allowed}])\**\s*[\).:\-]\s+",
        rf"^\s*\**([{allowed}])\**\s*(?=</THINK>|<TOOL_CALL>|</ARG_KEY>|<)",
        rf"(?:^|[\n\r])\s*\**([{allowed}])\**\s*(?=</THINK>|<TOOL_CALL>|</ARG_KEY>|<)",
        rf"^\s*\**([{allowed}])\**\s+(?=\S)",
        rf"^\s*\**([{allowed}])\**\s*[\).:\-]\s+",
        rf"^\s*([{allowed}])\s*[\).:\-]\s+",
        rf"(?:^|[\n\r])\s*\**([{allowed}])\**\s*(?=[\n\r]|$)",
        rf"(?:^|[\n\r])\s*\**([{allowed}])\**\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    standalone = [m.group(1) for m in CHOICE_RE.finditer(upper) if m.group(1) in labels]
    unique = sorted(set(standalone), key=standalone.index)
    if len(unique) == 1:
        return unique[0]
    return None


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]+", text.lower()))


REQUEST_FRAME_RE = re.compile(
    r"^\s*(?:the speaker is expressing the intention that\s+)?"
    r"(?:the speaker\s+)?wants\s+the\s+clinician\s+to\s+",
    re.IGNORECASE,
)

NEGATIVE_MARKER_RE = re.compile(
    r"\b("
    r"without|rather than|small talk|not needing|avoid explicit|without acting|"
    r"no further|not asking|not requesting|dismiss|close discussion|move past|"
    r"not want|does not|only"
    r")\b",
    re.IGNORECASE,
)

GENERIC_TEMPLATE_RE = re.compile(
    r"(review\s+.*\s+plan\s+follow-up|assess\s+risks\s+related\s+to|"
    r"generic\s+follow-up|generic\s+risk)",
    re.IGNORECASE,
)

DISTRACTOR_MARKER_PATTERNS = [
    r"related concern about",
    r"routine factual update",
    r"\bbefore\b",
    r"clearly today",
    r"carefully today",
    r"sensitively today",
    r"thoughtfully today",
    r"collaboratively today",
    r"respectfully today",
    r"care framing",
    r"plan priorities",
    r"practical care plan options",
]

DISTRACTOR_MARKER_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in DISTRACTOR_MARKER_PATTERNS),
    re.IGNORECASE,
)


def strip_request_frame(text: str) -> str:
    """Remove the controlled request frame before lexical artifact checks."""
    return REQUEST_FRAME_RE.sub("", text.strip())


def choice_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']*", strip_request_frame(text).lower())
    ngrams: set[str] = set()
    for n in range(min_n, max_n + 1):
        for idx in range(0, max(0, len(words) - n + 1)):
            ngrams.add(" ".join(words[idx : idx + n]))
    return ngrams


def mined_distractor_ngrams(tasks: Sequence[Dict[str, Any]]) -> List[str]:
    """Find corpus-level n-grams that appear disproportionately in distractors."""
    gold_counts: Counter[str] = Counter()
    distractor_counts: Counter[str] = Counter()
    for task in tasks:
        for idx, choice in enumerate(task["choices"]):
            target = gold_counts if idx == task["correct_idx"] else distractor_counts
            target.update(choice_ngrams(choice))

    min_count = max(15, int(round(0.03 * len(tasks))))
    mined = []
    for ngram, distractor_count in distractor_counts.items():
        gold_count = gold_counts.get(ngram, 0)
        if distractor_count >= min_count and distractor_count >= max(3 * gold_count, gold_count + min_count):
            mined.append((ngram, distractor_count, gold_count))
    mined.sort(key=lambda row: (-row[1], row[2], row[0]))
    return [ngram for ngram, _, _ in mined[:50]]


def extract_patient_utterance(prompt: str) -> str:
    match = re.search(r'Speaker says:\s*"(.*?)"', prompt, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'Patient says:\s*"(.*?)"', prompt, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'Speaker:\s*"(.*?)"', prompt, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'Patient:\s*"(.*?)"', prompt, re.DOTALL)
    if match:
        return match.group(1)
    return split_multiple_choice_prompt(prompt)


def lexical_overlap_baseline(
    tasks: Sequence[Dict[str, Any]],
    source: str = "utterance",
) -> Dict[str, Any]:
    """Compute a non-leaky lexical baseline.

    source="utterance" uses only the speaker utterance.
    source="context_utterance" uses everything before the answer choices.
    """
    results = []
    for task in tasks:
        if source == "context_utterance":
            source_text = split_multiple_choice_prompt(task["prompt"])
        elif source == "utterance":
            source_text = extract_patient_utterance(task["prompt"])
        else:
            raise ValueError(f"Unknown lexical source: {source}")

        source_words = token_set(source_text)
        scores = [len(source_words & token_set(choice)) for choice in task["choices"]]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        answer = chr(ord("A") + best_idx)
        results.append(
            {
                "eval_id": task.get("eval_id", task["probe_id"]),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "phenomena": task.get("phenomena", []),
                "model_answer": answer,
                "correct_letter": task["correct_letter"],
                "correct": best_idx == task["correct_idx"],
                "parse_failed": False,
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
            }
        )
    return {"metrics": compute_metrics(results), "results": results}


def lexical_overlap_after_request_frame_filter_baseline(
    tasks: Sequence[Dict[str, Any]],
    source: str = "utterance",
) -> Dict[str, Any]:
    """Lexical baseline after removing the shared request frame from choices."""
    results = []
    for task in tasks:
        if source == "context_utterance":
            source_text = split_multiple_choice_prompt(task["prompt"])
        elif source == "utterance":
            source_text = extract_patient_utterance(task["prompt"])
        else:
            raise ValueError(f"Unknown lexical source: {source}")

        source_words = token_set(source_text)
        scores = [len(source_words & token_set(strip_request_frame(choice))) for choice in task["choices"]]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        answer = chr(ord("A") + best_idx)
        results.append(
            {
                "eval_id": task.get("eval_id", task["probe_id"]),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "phenomena": task.get("phenomena", []),
                "model_answer": answer,
                "correct_letter": task["correct_letter"],
                "correct": best_idx == task["correct_idx"],
                "parse_failed": False,
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
                "request_frame_filter": True,
            }
        )
    return {"metrics": compute_metrics(results), "results": results}


def negative_marker_lexical_overlap_baseline(
    tasks: Sequence[Dict[str, Any]],
    source: str = "context_utterance",
) -> Dict[str, Any]:
    """Baseline that first exploits negative markers, then lexical overlap."""
    results = []
    for task in tasks:
        if source == "context_utterance":
            source_text = split_multiple_choice_prompt(task["prompt"])
        elif source == "utterance":
            source_text = extract_patient_utterance(task["prompt"])
        else:
            raise ValueError(f"Unknown lexical source: {source}")

        source_words = token_set(source_text)
        scores = [
            (
                int(bool(NEGATIVE_MARKER_RE.search(choice))),
                len(source_words & token_set(strip_request_frame(choice))),
            )
            for choice in task["choices"]
        ]
        best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        answer = chr(ord("A") + best_idx)
        results.append(
            {
                "eval_id": task.get("eval_id", task["probe_id"]),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "phenomena": task.get("phenomena", []),
                "model_answer": answer,
                "correct_letter": task["correct_letter"],
                "correct": best_idx == task["correct_idx"],
                "parse_failed": False,
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
                "negative_marker_lexical_scores": scores,
            }
        )
    return {"metrics": compute_metrics(results), "results": results}


def generic_template_marker_lexical_overlap_baseline(
    tasks: Sequence[Dict[str, Any]],
    source: str = "context_utterance",
) -> Dict[str, Any]:
    """Filter fixed generic templates as likely distractors, then use lexical overlap."""
    results = []
    for task in tasks:
        if source == "context_utterance":
            source_text = split_multiple_choice_prompt(task["prompt"])
        elif source == "utterance":
            source_text = extract_patient_utterance(task["prompt"])
        else:
            raise ValueError(f"Unknown lexical source: {source}")

        source_words = token_set(source_text)
        generic_flags = [bool(GENERIC_TEMPLATE_RE.search(choice)) for choice in task["choices"]]
        candidate_indices = [idx for idx, is_generic in enumerate(generic_flags) if not is_generic]
        if not candidate_indices:
            candidate_indices = list(range(len(task["choices"])))

        scores = [
            len(source_words & token_set(strip_request_frame(task["choices"][idx])))
            for idx in candidate_indices
        ]
        best_local_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -candidate_indices[idx]))
        best_idx = candidate_indices[best_local_idx]
        answer = chr(ord("A") + best_idx)
        results.append(
            {
                "eval_id": task.get("eval_id", task["probe_id"]),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "phenomena": task.get("phenomena", []),
                "model_answer": answer,
                "correct_letter": task["correct_letter"],
                "correct": best_idx == task["correct_idx"],
                "parse_failed": False,
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
                "generic_template_flags": generic_flags,
                "candidate_indices_after_generic_filter": candidate_indices,
                "generic_template_filtered_lexical_scores": scores,
            }
        )
    return {"metrics": compute_metrics(results), "results": results}


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _char_count(text: str) -> int:
    return len(text)


def _terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if stripped and stripped[-1] in ".!?;:":
        return stripped[-1]
    return "NONE"


def _starts_with(text: str, n_words: int = 4) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return " ".join(words[:n_words])


def _style_signature(text: str) -> tuple[str, str, str, bool, bool]:
    stripped = text.strip()
    return (
        _starts_with(stripped, n_words=4),
        _terminal_punctuation(stripped),
        "period" if stripped.endswith(".") else "non_period",
        stripped.lower().startswith("the patient"),
        stripped.lower().startswith("patient"),
    )


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def length_artifact_profile(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gold_chars = []
    distractor_chars = []
    gold_words = []
    distractor_words = []
    gold_shortest = 0
    gold_longest = 0
    gold_word_shortest = 0
    gold_word_longest = 0
    gold_strict_shortest = 0
    gold_strict_longest = 0
    gold_strict_word_shortest = 0
    gold_strict_word_longest = 0
    for task in tasks:
        choices = list(task["choices"])
        correct_idx = task["correct_idx"]
        char_lengths = [_char_count(choice) for choice in choices]
        word_lengths = [_word_count(choice) for choice in choices]
        gold_chars.append(char_lengths[correct_idx])
        gold_words.append(word_lengths[correct_idx])
        distractor_chars.extend(
            length for idx, length in enumerate(char_lengths) if idx != correct_idx
        )
        distractor_words.extend(
            length for idx, length in enumerate(word_lengths) if idx != correct_idx
        )
        min_chars = min(char_lengths)
        max_chars = max(char_lengths)
        min_words = min(word_lengths)
        max_words = max(word_lengths)
        gold_shortest += int(char_lengths[correct_idx] == min_chars)
        gold_longest += int(char_lengths[correct_idx] == max_chars)
        gold_word_shortest += int(word_lengths[correct_idx] == min_words)
        gold_word_longest += int(word_lengths[correct_idx] == max_words)
        gold_strict_shortest += int(
            char_lengths[correct_idx] == min_chars and char_lengths.count(min_chars) == 1
        )
        gold_strict_longest += int(
            char_lengths[correct_idx] == max_chars and char_lengths.count(max_chars) == 1
        )
        gold_strict_word_shortest += int(
            word_lengths[correct_idx] == min_words and word_lengths.count(min_words) == 1
        )
        gold_strict_word_longest += int(
            word_lengths[correct_idx] == max_words and word_lengths.count(max_words) == 1
        )

    total = len(tasks)
    mean_gold_chars = sum(gold_chars) / len(gold_chars) if gold_chars else 0.0
    mean_distractor_chars = (
        sum(distractor_chars) / len(distractor_chars) if distractor_chars else 0.0
    )
    mean_gold_words = sum(gold_words) / len(gold_words) if gold_words else 0.0
    mean_distractor_words = (
        sum(distractor_words) / len(distractor_words) if distractor_words else 0.0
    )
    return {
        "gold_is_shortest_share": gold_shortest / total if total else 0.0,
        "gold_is_longest_share": gold_longest / total if total else 0.0,
        "gold_is_shortest_word_share": gold_word_shortest / total if total else 0.0,
        "gold_is_longest_word_share": gold_word_longest / total if total else 0.0,
        "gold_is_strict_shortest_share": gold_strict_shortest / total if total else 0.0,
        "gold_is_strict_longest_share": gold_strict_longest / total if total else 0.0,
        "gold_is_strict_shortest_word_share": gold_strict_word_shortest / total if total else 0.0,
        "gold_is_strict_longest_word_share": gold_strict_word_longest / total if total else 0.0,
        "mean_gold_chars": mean_gold_chars,
        "mean_distractor_chars": mean_distractor_chars,
        "mean_gold_minus_distractor_chars": mean_gold_chars - mean_distractor_chars,
        "mean_gold_words": mean_gold_words,
        "mean_distractor_words": mean_distractor_words,
        "mean_gold_minus_distractor_words": mean_gold_words - mean_distractor_words,
    }


def artifact_feature_baseline(
    tasks: Sequence[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    """Evaluate a non-semantic answer-choice artifact baseline.

    These baselines intentionally use the corpus-level gold feature profile.
    They do not inspect clinical context or utterance content. Their purpose is
    to detect exploitable answer-choice artifacts after dataset construction.
    """

    if mode not in {
        "style_only",
        "length_only",
        "punctuation_only",
        "starts_with",
        "contains_wants_clinician",
        "request_frame",
        "negative_marker",
        "generic_template_marker",
        "distractor_marker_baseline",
        "marker_filter_plus_shortest_baseline",
        "mined_distractor_ngram_baseline",
        "shortest_char_baseline",
        "longest_char_baseline",
        "shortest_word_baseline",
        "longest_word_baseline",
    }:
        raise ValueError(f"Unknown artifact baseline mode: {mode}")

    gold_choices = [task["choices"][task["correct_idx"]] for task in tasks]
    gold_lengths = [_word_count(choice) for choice in gold_choices]
    median_gold_length = _median(gold_lengths)
    common_punctuation = Counter(_terminal_punctuation(choice) for choice in gold_choices).most_common(1)
    common_start = Counter(_starts_with(choice, n_words=4) for choice in gold_choices).most_common(1)
    common_style = Counter(_style_signature(choice) for choice in gold_choices).most_common(1)
    punctuation_target = common_punctuation[0][0] if common_punctuation else "NONE"
    start_target = common_start[0][0] if common_start else ""
    style_target = common_style[0][0] if common_style else ("", "NONE", "non_period", False, False)
    mined_ngrams = mined_distractor_ngrams(tasks)

    def score(choice: str) -> float:
        lowered = choice.lower()
        if mode == "contains_wants_clinician":
            return float("wants the clinician" in lowered)
        if mode == "request_frame":
            return float(bool(REQUEST_FRAME_RE.search(choice)))
        if mode == "negative_marker":
            return float(bool(NEGATIVE_MARKER_RE.search(choice)))
        if mode == "generic_template_marker":
            return float(not bool(GENERIC_TEMPLATE_RE.search(choice)))
        if mode == "distractor_marker_baseline":
            return float(not bool(DISTRACTOR_MARKER_RE.search(choice)))
        if mode == "marker_filter_plus_shortest_baseline":
            marker_penalty = 100000 if DISTRACTOR_MARKER_RE.search(choice) else 0
            return -(marker_penalty + _char_count(choice))
        if mode == "mined_distractor_ngram_baseline":
            ngrams = choice_ngrams(choice)
            return -sum(1 for ngram in mined_ngrams if ngram in ngrams)
        if mode == "length_only":
            return -abs(_word_count(choice) - median_gold_length)
        if mode == "shortest_char_baseline":
            return -_char_count(choice)
        if mode == "longest_char_baseline":
            return _char_count(choice)
        if mode == "shortest_word_baseline":
            return -_word_count(choice)
        if mode == "longest_word_baseline":
            return _word_count(choice)
        if mode == "punctuation_only":
            return float(_terminal_punctuation(choice) == punctuation_target)
        if mode == "starts_with":
            return float(_starts_with(choice, n_words=4) == start_target)
        signature = _style_signature(choice)
        return float(sum(1 for got, want in zip(signature, style_target) if got == want))

    results = []
    for task in tasks:
        scores = [score(choice) for choice in task["choices"]]
        best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        answer = chr(ord("A") + best_idx)
        results.append(
            {
                "eval_id": task.get("eval_id", task["probe_id"]),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "phenomena": task.get("phenomena", []),
                "model_answer": answer,
                "correct_letter": task["correct_letter"],
                "correct": best_idx == task["correct_idx"],
                "parse_failed": False,
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
                "artifact_scores": scores,
            }
        )

    profile = {
        "mode": mode,
        "median_gold_word_count": median_gold_length,
        "most_common_gold_punctuation": punctuation_target,
        "most_common_gold_start_4_words": start_target,
        "most_common_gold_style_signature": list(style_target),
        "request_frame": "the speaker wants the clinician to",
        "generic_template_patterns": [
            "review .* plan follow-up",
            "assess risks related to",
            "generic follow-up",
            "generic risk",
        ],
        "distractor_marker_patterns": DISTRACTOR_MARKER_PATTERNS,
        "mined_distractor_ngrams": mined_ngrams,
        "length_artifacts": length_artifact_profile(tasks),
    }
    return {"metrics": compute_metrics(results), "profile": profile, "results": results}


def artifact_baselines(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        mode: artifact_feature_baseline(tasks, mode)
        for mode in [
            "style_only",
            "length_only",
            "punctuation_only",
            "starts_with",
            "contains_wants_clinician",
            "request_frame",
            "negative_marker",
            "generic_template_marker",
            "distractor_marker_baseline",
            "marker_filter_plus_shortest_baseline",
            "mined_distractor_ngram_baseline",
            "shortest_char_baseline",
            "longest_char_baseline",
            "shortest_word_baseline",
            "longest_word_baseline",
        ]
    }


def majority_vote_by_probe(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate repeated option-order runs by probe using majority vote."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["probe_id"]].append(result)

    aggregated = []
    for probe_id, probe_results in grouped.items():
        first = probe_results[0]
        any_phenomenon_correct = None

        if any("model_choice" in r for r in probe_results):
            valid_predictions = [
                r["model_choice"]
                for r in probe_results
                if r.get("model_choice") and not r.get("parse_failed")
            ]
            counts = Counter(valid_predictions)
            if counts:
                prediction, votes = counts.most_common(1)[0]
            else:
                prediction, votes = None, 0
            correct = prediction == first.get("recognition_gold_choice")
            model_answer = prediction
        elif any(
            "predicted_phenomenon" in r or "correct_letters" in r for r in probe_results
        ):
            valid_predictions = [
                r["predicted_phenomenon"]
                for r in probe_results
                if r.get("predicted_phenomenon") and not r.get("parse_failed")
            ]
            counts = Counter(valid_predictions)
            if counts:
                prediction, votes = counts.most_common(1)[0]
            else:
                prediction, votes = None, 0
            primary_gold = first.get("classification_primary_gold")
            any_gold_codes = set(
                first.get("classification_any_gold_codes")
                or first.get("classification_gold_codes")
                or first.get("phenomena", [])
            )
            correct = prediction is not None and prediction == primary_gold
            any_phenomenon_correct = prediction is not None and prediction in any_gold_codes
            model_answer = prediction
        else:
            valid_answers = [
                r["model_answer"]
                for r in probe_results
                if r.get("model_answer") and not r.get("parse_failed")
            ]
            counts = Counter(valid_answers)
            if counts:
                model_answer, votes = counts.most_common(1)[0]
            else:
                model_answer, votes = None, 0
            correct = (
                model_answer is not None
                and first.get("correct_letter") is not None
                and model_answer == first.get("correct_letter")
            )

        aggregated.append(
            {
                "probe_id": probe_id,
                "domain": first.get("domain"),
                "variant": first.get("variant"),
                "phenomena": first.get("phenomena", []),
                "classification_primary_gold": first.get("classification_primary_gold"),
                "classification_gold_codes": first.get("classification_gold_codes"),
                "classification_any_gold_codes": first.get("classification_any_gold_codes"),
                "model_answer": model_answer,
                "correct_letter": first.get("correct_letter"),
                "recognition_gold_choice": first.get("recognition_gold_choice"),
                "correct": correct,
                **(
                    {"any_phenomenon_correct": any_phenomenon_correct}
                    if any_phenomenon_correct is not None
                    else {}
                ),
                "votes": votes,
                "runs": len(probe_results),
                "parse_failed": model_answer is None,
            }
        )
    return aggregated


def compute_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    metrics: Dict[str, Any] = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_failures": sum(1 for r in results if r.get("parse_failed")),
    }
    metrics["parse_failure_rate"] = metrics["parse_failures"] / total if total else 0.0
    if any("any_phenomenon_correct" in r for r in results):
        any_total = sum(1 for r in results if "any_phenomenon_correct" in r)
        any_correct = sum(1 for r in results if r.get("any_phenomenon_correct"))
        metrics["any_phenomenon_total"] = any_total
        metrics["any_phenomenon_correct"] = any_correct
        metrics["any_phenomenon_accuracy"] = any_correct / any_total if any_total else 0.0

    answer_counts = Counter(r.get("model_answer") or "?" for r in results)
    metrics["answer_distribution"] = dict(answer_counts)
    metrics["max_position_share"] = max(answer_counts.values()) / total if total else 0.0

    def add_group(prefix: str, key_fn: Callable[[Dict[str, Any]], Iterable[str] | str | None]) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            keys = key_fn(result)
            if keys is None:
                continue
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                grouped[key].append(result)
        for key, subset in grouped.items():
            if not subset:
                continue
            metrics[f"{prefix}_{key}"] = sum(1 for r in subset if r.get("correct")) / len(subset)
            metrics[f"{prefix}_{key}_n"] = len(subset)

    add_group("accuracy", lambda r: r.get("domain"))
    add_group("accuracy_variant", lambda r: r.get("variant"))
    add_group("accuracy_phen", lambda r: r.get("phenomena", []))

    direct = [r for r in results if r.get("variant") == "direct"]
    pragmatic = [r for r in results if r.get("variant") != "direct"]
    if direct:
        metrics["accuracy_variant_direct"] = sum(1 for r in direct if r.get("correct")) / len(direct)
        metrics["accuracy_direct_n"] = len(direct)
    if pragmatic:
        metrics["accuracy_variant_pragmatic"] = sum(1 for r in pragmatic if r.get("correct")) / len(pragmatic)
        metrics["accuracy_pragmatic_n"] = len(pragmatic)
    if direct and pragmatic:
        metrics["pragmatic_gap"] = (
            metrics["accuracy_variant_direct"] - metrics["accuracy_variant_pragmatic"]
        )
    return metrics


def estimate_prompt_tokens(prompts: Sequence[str], multiplier: float = 1.0) -> int:
    """Conservative tokenizer-agnostic estimate."""
    return int(round(sum(len(prompt) / 4 for prompt in prompts) * multiplier))


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
