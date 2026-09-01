#!/usr/bin/env python3
"""Summarize final sign-off-filtered model, classification, and generation-judge results.

The final table is item-level only: a model row is reported only when the
workspace contains raw per-item outputs that can be filtered to the final mask.
Legacy summary-only API rows are intentionally not reused for final-release
counts because they cannot be reweighted after expert sign-off exclusions.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import (  # noqa: E402
    artifact_baselines,
    compute_metrics,
    lexical_overlap_baseline,
    load_compact_tasks,
    majority_vote_by_probe,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
VALIDATION_DIR = ROOT / "data" / "validation"
EXPERIMENT_DIR = ROOT / "experiments"
OUT_DIR = EXPERIMENT_DIR / "model_results"

CORE_PATH = PROCESSED_DIR / "pcp_core_3000_final.json"
HARD_PATH = PROCESSED_DIR / "pcp_hard_469_signoff_final.json"
REAL_PATH = PROCESSED_DIR / "pcp_real_clean_64_signoff_final.json"
MASK_PATH = VALIDATION_DIR / "final_inclusion_mask.csv"
API_RESULT_DIRS = {
    "core": EXPERIMENT_DIR / "api_core_3000_final",
    "hard": EXPERIMENT_DIR / "api_hard_469_final",
    "real_clean": EXPERIMENT_DIR / "api_real_clean_64_final",
}
SUMMARY_JSON = OUT_DIR / "final_model_results_after_signoff_summary.json"
SUMMARY_CSV = OUT_DIR / "final_model_results_after_signoff.csv"
TABLE_TEX = ROOT / "paper" / "tables" / "results.tex"
CLASSIFICATION_JSON = OUT_DIR / "classification_final_results.json"
CLASSIFICATION_CSV = OUT_DIR / "classification_final_results.csv"
CLASSIFICATION_TEX = ROOT / "paper" / "tables" / "classification_final_results.tex"
GENERATION_JUDGE_JSON = OUT_DIR / "generation_judge_results.json"
GENERATION_JUDGE_CSV = OUT_DIR / "generation_judge_results.csv"
GENERATION_JUDGE_TEX = ROOT / "paper" / "tables" / "generation_judge_results.tex"
GENERATION_JUDGE_CONSENSUS_JSON = OUT_DIR / "generation_judge_consensus_3judge.json"

CSV_COLUMNS = [
    "model",
    "type",
    "source",
    "core_n",
    "core_accuracy",
    "hard_n",
    "hard_accuracy",
    "hard_pragmatic_n",
    "hard_pragmatic_accuracy",
    "hard_direct_n",
    "hard_direct_accuracy",
    "real_clean_n",
    "real_clean_accuracy",
    "aggregate_n",
    "aggregate_accuracy",
    "parse_failure_rate",
    "notes",
]

CLASSIFICATION_CSV_COLUMNS = [
    "model",
    "type",
    "source",
    "core_n",
    "core_primary_accuracy",
    "core_any_phenomenon_accuracy",
    "hard_n",
    "hard_primary_accuracy",
    "hard_any_phenomenon_accuracy",
    "hard_pragmatic_n",
    "hard_pragmatic_primary_accuracy",
    "hard_pragmatic_any_phenomenon_accuracy",
    "hard_direct_n",
    "hard_direct_primary_accuracy",
    "hard_direct_any_phenomenon_accuracy",
    "real_clean_n",
    "real_clean_primary_accuracy",
    "real_clean_any_phenomenon_accuracy",
    "aggregate_n",
    "aggregate_primary_accuracy",
    "aggregate_any_phenomenon_accuracy",
    "parse_failure_rate",
    "notes",
]

GENERATION_JUDGE_CSV_COLUMNS = [
    "model",
    "judge_models",
    "source_files",
    "core_n",
    "core_overall_success_rate",
    "core_overall_success_rate_std",
    "core_intent_match_mean",
    "core_clinical_action_match_mean",
    "core_pragmatic_cue_capture_mean",
    "core_unsupported_inference_rate",
    "core_unsafe_or_misleading_rate",
    "hard_n",
    "hard_overall_success_rate",
    "hard_overall_success_rate_std",
    "hard_intent_match_mean",
    "hard_clinical_action_match_mean",
    "hard_pragmatic_cue_capture_mean",
    "hard_unsupported_inference_rate",
    "hard_unsafe_or_misleading_rate",
    "aggregate_n",
    "aggregate_overall_success_rate",
    "aggregate_overall_success_rate_std",
    "aggregate_intent_match_mean",
    "aggregate_clinical_action_match_mean",
    "aggregate_pragmatic_cue_capture_mean",
    "aggregate_unsupported_inference_rate",
    "aggregate_unsafe_or_misleading_rate",
    "aggregate_pairwise_cohens_kappa_mean",
    "pairwise_cohens_kappa",
    "drop_self_ablation",
    "notes",
]

LB = r"\\"
MODEL_DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "claude-opus-4-7": "Claude Opus 4.7",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "mimo-v2.5-pro": "MiMo-V2.5-Pro",
    "MiniMax-M2.7": "MiniMax-M2.7",
    "minimaxai/minimax-m2.7": "MiniMax-M2.7",
    "minimax-m2.7": "MiniMax-M2.7",
    "z-ai/glm-5.1": "GLM-5.1",
    "glm-5.1": "GLM-5.1",
    "google/medgemma-27b-it": "MedGemma-27B-it",
    "medgemma-27b-it": "MedGemma-27B-it",
    "google_medgemma-27b-it": "MedGemma-27B-it",
    "BioMistral/BioMistral-7B": "BioMistral-7B",
    "biomistral-7b": "BioMistral-7B",
    "BioMistral_BioMistral-7B": "BioMistral-7B",
    "google/gemma-3-27b-it": "Gemma-3-27B-it",
    "gemma-3-27b-it": "Gemma-3-27B-it",
    "google_gemma-3-27b-it": "Gemma-3-27B-it",
    "google/gemma-4-31B-it": "Gemma-4-31B-it",
    "gemma-4-31B-it": "Gemma-4-31B-it",
    "gemma-4-31b-it": "Gemma-4-31B-it",
    "google_gemma-4-31B-it": "Gemma-4-31B-it",
    "google_gemma-4-31b-it": "Gemma-4-31B-it",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral-Small-3.2-24B-Instruct",
    "Mistral-Small-3.2-24B-Instruct": "Mistral-Small-3.2-24B-Instruct",
    "mistralai_Mistral-Small-3.2-24B-Instruct-2506": "Mistral-Small-3.2-24B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instruct",
    "Qwen3-30B-A3B-Instruct": "Qwen3-30B-A3B-Instruct",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instruct",
    "grok-4.3": "Grok 4.3",
    "x-ai/grok-4.3": "Grok 4.3",
    "mistral-medium-2508": "Mistral Medium 3.1",
    "mistralai/mistral-medium-3.1": "Mistral Medium 3.1",
}
MODEL_ORDER = {model: idx for idx, model in enumerate(MODEL_DISPLAY)}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    columns: Sequence[str] = CSV_COLUMNS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}"


def pct_pm(value: float | None, std: float | None) -> str:
    if value is None:
        return "--"
    if std in {None, ""}:
        return pct(value)
    return f"{100 * float(value):.1f}$\\pm${100 * float(std):.1f}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def accuracy_from_metrics(metrics: Dict[str, Any], key: str = "accuracy") -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def position_results(tasks: Sequence[Dict[str, Any]], answer: str = "A") -> List[Dict[str, Any]]:
    return [
        {
            "probe_id": task["probe_id"],
            "domain": task.get("domain"),
            "variant": task.get("variant"),
            "phenomena": task.get("phenomena", []),
            "model_answer": answer,
            "correct_letter": task["correct_letter"],
            "correct": answer == task["correct_letter"],
            "parse_failed": False,
        }
        for task in tasks
    ]


def row_template(model: str, row_type: str, source: str, notes: str = "") -> Dict[str, Any]:
    return {
        "model": model,
        "type": row_type,
        "source": source,
        "notes": notes,
    }


def fill_scope_metrics(
    row: Dict[str, Any],
    *,
    core: Dict[str, Any] | None,
    hard: Dict[str, Any] | None,
    real_clean: Dict[str, Any] | None,
    aggregate: Dict[str, Any] | None,
) -> None:
    if core:
        row["core_n"] = core.get("total")
        row["core_accuracy"] = accuracy_from_metrics(core)
    if hard:
        row["hard_n"] = hard.get("total")
        row["hard_accuracy"] = accuracy_from_metrics(hard)
        row["hard_pragmatic_n"] = hard.get("accuracy_pragmatic_n")
        row["hard_pragmatic_accuracy"] = accuracy_from_metrics(hard, "accuracy_variant_pragmatic")
        row["hard_direct_n"] = hard.get("accuracy_direct_n")
        row["hard_direct_accuracy"] = accuracy_from_metrics(hard, "accuracy_variant_direct")
    if real_clean:
        row["real_clean_n"] = real_clean.get("total")
        row["real_clean_accuracy"] = accuracy_from_metrics(real_clean)
    if aggregate:
        row["aggregate_n"] = aggregate.get("total")
        row["aggregate_accuracy"] = accuracy_from_metrics(aggregate)
        row["parse_failure_rate"] = aggregate.get("parse_failure_rate")
    elif hard:
        row["parse_failure_rate"] = hard.get("parse_failure_rate")


def baseline_rows(
    core_tasks: Sequence[Dict[str, Any]],
    hard_tasks: Sequence[Dict[str, Any]],
    real_tasks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregate_tasks = list(core_tasks) + list(hard_tasks) + list(real_tasks)
    rows: List[Dict[str, Any]] = []

    counts = {
        "core": len(core_tasks),
        "hard": len(hard_tasks),
        "hard_pragmatic": sum(1 for task in hard_tasks if task.get("variant") != "direct"),
        "hard_direct": sum(1 for task in hard_tasks if task.get("variant") == "direct"),
        "real_clean": len(real_tasks),
        "aggregate": len(aggregate_tasks),
    }
    random_row = row_template("Random", "Baseline", "Expected uniform random choice")
    random_row.update(
        {
            "core_n": counts["core"],
            "core_accuracy": 0.25,
            "hard_n": counts["hard"],
            "hard_accuracy": 0.25,
            "hard_pragmatic_n": counts["hard_pragmatic"],
            "hard_pragmatic_accuracy": 0.25,
            "hard_direct_n": counts["hard_direct"],
            "hard_direct_accuracy": 0.25,
            "real_clean_n": counts["real_clean"],
            "real_clean_accuracy": 0.25,
            "aggregate_n": counts["aggregate"],
            "aggregate_accuracy": 0.25,
            "parse_failure_rate": 0.0,
        }
    )
    rows.append(random_row)

    always_a = row_template("Always A", "Baseline", "Deterministic position baseline")
    fill_scope_metrics(
        always_a,
        core=compute_metrics(position_results(core_tasks)),
        hard=compute_metrics(position_results(hard_tasks)),
        real_clean=compute_metrics(position_results(real_tasks)),
        aggregate=compute_metrics(position_results(aggregate_tasks)),
    )
    rows.append(always_a)

    for display, source in [
        ("Lexical (utterance)", "utterance"),
        ("Lexical overlap (ctx+utt.)", "context_utterance"),
    ]:
        row = row_template(display, "Baseline", f"lexical_overlap:{source}")
        fill_scope_metrics(
            row,
            core=lexical_overlap_baseline(core_tasks, source)["metrics"],
            hard=lexical_overlap_baseline(hard_tasks, source)["metrics"],
            real_clean=lexical_overlap_baseline(real_tasks, source)["metrics"],
            aggregate=lexical_overlap_baseline(aggregate_tasks, source)["metrics"],
        )
        rows.append(row)

    artifact_labels = [
        ("Style-only artifact", "style_only"),
        ("Length-only artifact", "length_only"),
        ("Punctuation-only artifact", "punctuation_only"),
        ("Starts-with artifact", "starts_with"),
    ]
    core_artifacts = artifact_baselines(core_tasks)
    hard_artifacts = artifact_baselines(hard_tasks)
    real_artifacts = artifact_baselines(real_tasks)
    aggregate_artifacts = artifact_baselines(aggregate_tasks)
    for display, mode in artifact_labels:
        row = row_template(display, "Artifact", f"artifact_baseline:{mode}")
        fill_scope_metrics(
            row,
            core=core_artifacts[mode]["metrics"],
            hard=hard_artifacts[mode]["metrics"],
            real_clean=real_artifacts[mode]["metrics"],
            aggregate=aggregate_artifacts[mode]["metrics"],
        )
        rows.append(row)

    return rows


def model_display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.get(model, 99), model)


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def reusable_api_result(row: Dict[str, Any], group: str) -> bool:
    if row.get("error"):
        return False
    usage = row.get("usage", {}) or {}
    finish_reason = str(
        usage.get("finish_reason") or usage.get("stop_reason") or ""
    ).lower()
    truncated = finish_reason in {"length", "max_tokens", "max_output_tokens"}
    if group != "generation" and row.get("parse_failed") and truncated:
        return False
    if group != "generation" and row.get("parse_failed") and not row.get("raw_response"):
        return False
    if group == "generation" and (
        truncated or not (row.get("model_response") or "").strip()
    ):
        return False
    return True


def api_result_rank(row: Dict[str, Any], group: str) -> tuple[int, int]:
    if not reusable_api_result(row, group):
        return (0, 0)
    return (2 if not row.get("parse_failed") else 1, int(row.get("correct") is True))


def dedupe_api_results(rows: Sequence[Dict[str, Any]], group: str) -> List[Dict[str, Any]]:
    by_eval_id: Dict[str, Dict[str, Any]] = {}
    ranks: Dict[str, tuple[int, int]] = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if not eval_id:
            continue
        rank = api_result_rank(row, group)
        if eval_id not in by_eval_id or rank > ranks[eval_id]:
            by_eval_id[eval_id] = row
            ranks[eval_id] = rank
    return [by_eval_id[key] for key in sorted(by_eval_id)]


def load_api_result_rows(expected_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "metrics": {},
            "rows": {},
            "raw_rows": defaultdict(list),
            "sources": [],
            "notes": [],
        }
    )

    for split, input_dir in API_RESULT_DIRS.items():
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.glob("*_pcp_full.json")):
            payload = read_json(path)
            model = payload.get("model") or path.stem
            recognition = payload.get("results", {}).get("recognition", [])
            entry = by_model[model]
            entry["sources"].append(rel(path))
            entry["raw_rows"][split].extend(recognition)

    out: List[Dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        entry = by_model[model]
        for split, raw_rows in entry["raw_rows"].items():
            recognition = dedupe_api_results(raw_rows, "recognition")
            majority_rows = majority_vote_by_probe(recognition) if recognition else []
            metrics = compute_metrics(majority_rows) if majority_rows else {}
            total = int(metrics.get("total", 0) or 0)
            expected = expected_counts[split]
            if total != expected:
                entry["notes"].append(
                    f"{split} incomplete: expected {expected} final items, got {total}"
                )
                continue
            entry["metrics"][split] = metrics
            entry["rows"][split] = majority_rows
        if not entry["metrics"]:
            continue
        aggregate_metrics = None
        if all(split in entry["rows"] for split in ["core", "hard", "real_clean"]):
            aggregate_rows = (
                entry["rows"]["core"]
                + entry["rows"]["hard"]
                + entry["rows"]["real_clean"]
            )
            aggregate_metrics = compute_metrics(aggregate_rows)

        row = row_template(
            model_display(model),
            "Model",
            " ; ".join(entry["sources"]),
            notes=" ; ".join(entry["notes"]),
        )
        fill_scope_metrics(
            row,
            core=entry["metrics"].get("core"),
            hard=entry["metrics"].get("hard"),
            real_clean=entry["metrics"].get("real_clean"),
            aggregate=aggregate_metrics,
        )
        out.append(row)
    return out


def any_variant_accuracy(rows: Sequence[Dict[str, Any]], *, direct: bool) -> float | None:
    subset = [
        row for row in rows
        if ("any_phenomenon_correct" in row)
        and ((row.get("variant") == "direct") == direct)
    ]
    if not subset:
        return None
    return sum(1 for row in subset if row.get("any_phenomenon_correct")) / len(subset)


def classification_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = compute_metrics(rows)
    pragmatic_any = any_variant_accuracy(rows, direct=False)
    direct_any = any_variant_accuracy(rows, direct=True)
    if pragmatic_any is not None:
        metrics["any_phenomenon_accuracy_variant_pragmatic"] = pragmatic_any
    if direct_any is not None:
        metrics["any_phenomenon_accuracy_variant_direct"] = direct_any
    return metrics


def fill_classification_metrics(
    row: Dict[str, Any],
    *,
    core: Dict[str, Any] | None,
    hard: Dict[str, Any] | None,
    real_clean: Dict[str, Any] | None,
    aggregate: Dict[str, Any] | None,
) -> None:
    if core:
        row["core_n"] = core.get("total")
        row["core_primary_accuracy"] = accuracy_from_metrics(core)
        row["core_any_phenomenon_accuracy"] = accuracy_from_metrics(
            core, "any_phenomenon_accuracy"
        )
    if hard:
        row["hard_n"] = hard.get("total")
        row["hard_primary_accuracy"] = accuracy_from_metrics(hard)
        row["hard_any_phenomenon_accuracy"] = accuracy_from_metrics(
            hard, "any_phenomenon_accuracy"
        )
        row["hard_pragmatic_n"] = hard.get("accuracy_pragmatic_n")
        row["hard_pragmatic_primary_accuracy"] = accuracy_from_metrics(
            hard, "accuracy_variant_pragmatic"
        )
        row["hard_pragmatic_any_phenomenon_accuracy"] = accuracy_from_metrics(
            hard, "any_phenomenon_accuracy_variant_pragmatic"
        )
        row["hard_direct_n"] = hard.get("accuracy_direct_n")
        row["hard_direct_primary_accuracy"] = accuracy_from_metrics(
            hard, "accuracy_variant_direct"
        )
        row["hard_direct_any_phenomenon_accuracy"] = accuracy_from_metrics(
            hard, "any_phenomenon_accuracy_variant_direct"
        )
        row["parse_failure_rate"] = hard.get("parse_failure_rate")
    if real_clean:
        row["real_clean_n"] = real_clean.get("total")
        row["real_clean_primary_accuracy"] = accuracy_from_metrics(real_clean)
        row["real_clean_any_phenomenon_accuracy"] = accuracy_from_metrics(
            real_clean, "any_phenomenon_accuracy"
        )
    if aggregate:
        row["aggregate_n"] = aggregate.get("total")
        row["aggregate_primary_accuracy"] = accuracy_from_metrics(aggregate)
        row["aggregate_any_phenomenon_accuracy"] = accuracy_from_metrics(
            aggregate, "any_phenomenon_accuracy"
        )
        row["parse_failure_rate"] = aggregate.get("parse_failure_rate")


def load_api_classification_rows(expected_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "metrics": {},
            "rows": {},
            "raw_rows": defaultdict(list),
            "sources": [],
            "notes": [],
        }
    )

    for split, input_dir in API_RESULT_DIRS.items():
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.glob("*_pcp_full.json")):
            payload = read_json(path)
            model = payload.get("model") or path.stem
            classification = payload.get("results", {}).get("classification", [])
            entry = by_model[model]
            entry["sources"].append(rel(path))
            entry["raw_rows"][split].extend(classification)

    out: List[Dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        entry = by_model[model]
        for split, raw_rows in entry["raw_rows"].items():
            classification = dedupe_api_results(raw_rows, "classification")
            majority_rows = majority_vote_by_probe(classification) if classification else []
            metrics = classification_metrics(majority_rows) if majority_rows else {}
            total = int(metrics.get("total", 0) or 0)
            expected = expected_counts[split]
            if total != expected:
                entry["notes"].append(
                    f"{split} incomplete: expected {expected} final items, got {total}"
                )
                continue
            entry["metrics"][split] = metrics
            entry["rows"][split] = majority_rows
        if not entry["metrics"]:
            continue
        aggregate_metrics = None
        if all(split in entry["rows"] for split in ["core", "hard", "real_clean"]):
            aggregate_rows = (
                entry["rows"]["core"]
                + entry["rows"]["hard"]
                + entry["rows"]["real_clean"]
            )
            aggregate_metrics = classification_metrics(aggregate_rows)

        row = row_template(
            model_display(model),
            "Model",
            " ; ".join(entry["sources"]),
            notes=" ; ".join(entry["notes"]),
        )
        fill_classification_metrics(
            row,
            core=entry["metrics"].get("core"),
            hard=entry["metrics"].get("hard"),
            real_clean=entry["metrics"].get("real_clean"),
            aggregate=aggregate_metrics,
        )
        out.append(row)
    return out


def missing_raw_outputs() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    for split, path in API_RESULT_DIRS.items():
        files = sorted(path.glob("*_pcp_full.json")) if path.exists() else []
        status[path.name] = {
            "split": split,
            "path": rel(path),
            "exists": path.exists(),
            "raw_json_files": [rel(file) for file in files],
            "reported": bool(files),
        }
    return status


def generation_judge_files() -> List[Path]:
    if not OUT_DIR.exists():
        return []
    files: List[Path] = []
    if GENERATION_JUDGE_CONSENSUS_JSON.exists():
        files.append(GENERATION_JUDGE_CONSENSUS_JSON)
    for path in sorted(OUT_DIR.glob("generation_judge_consensus*.json")):
        if path not in files:
            files.append(path)
    return files


def generation_split_key(split_name: str | None) -> str | None:
    value = split_name or ""
    if "Core" in value:
        return "core"
    if "Hard" in value:
        return "hard"
    return None


def valid_generation_judgment(row: Dict[str, Any]) -> bool:
    return not row.get("error") and not row.get("parse_failed")


def generation_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if valid_generation_judgment(row)]
    metrics: Dict[str, Any] = {
        "total": len(valid),
        "raw_total": len(rows),
        "parse_failures": sum(1 for row in rows if row.get("parse_failed")),
        "errors": sum(1 for row in rows if row.get("error")),
    }
    for field in [
        "intent_match_1_5",
        "clinical_action_match_1_5",
        "pragmatic_cue_capture_1_5",
    ]:
        metrics[f"{field}_mean"] = (
            sum(float(row[field]) for row in valid) / len(valid) if valid else None
        )
    for field in ["unsupported_inference", "unsafe_or_misleading", "overall_success"]:
        metrics[f"{field}_rate"] = (
            sum(1 for row in valid if row.get(field) == "yes") / len(valid) if valid else None
        )
    return metrics


def fill_generation_metrics(
    row: Dict[str, Any],
    *,
    prefix: str,
    metrics: Dict[str, Any] | None,
) -> None:
    if not metrics:
        return
    row[f"{prefix}_n"] = metrics.get("total")
    row[f"{prefix}_overall_success_rate"] = metrics.get("overall_success_rate")
    row[f"{prefix}_overall_success_rate_std"] = metrics.get("overall_success_rate_std")
    row[f"{prefix}_intent_match_mean"] = metrics.get("intent_match_1_5_mean")
    row[f"{prefix}_clinical_action_match_mean"] = metrics.get(
        "clinical_action_match_1_5_mean"
    )
    row[f"{prefix}_pragmatic_cue_capture_mean"] = metrics.get(
        "pragmatic_cue_capture_1_5_mean"
    )
    row[f"{prefix}_unsupported_inference_rate"] = metrics.get(
        "unsupported_inference_rate"
    )
    row[f"{prefix}_unsafe_or_misleading_rate"] = metrics.get(
        "unsafe_or_misleading_rate"
    )


def fill_generation_ensemble_metrics(
    row: Dict[str, Any],
    *,
    prefix: str,
    metrics: Dict[str, Any] | None,
) -> None:
    if not metrics:
        return
    row[f"{prefix}_n"] = metrics.get("n_source_items")
    row[f"{prefix}_overall_success_rate"] = metrics.get("overall_success_rate_mean")
    row[f"{prefix}_overall_success_rate_std"] = metrics.get("overall_success_rate_std")
    row[f"{prefix}_intent_match_mean"] = metrics.get("intent_match_1_5_mean")
    row[f"{prefix}_clinical_action_match_mean"] = metrics.get(
        "clinical_action_match_1_5_mean"
    )
    row[f"{prefix}_pragmatic_cue_capture_mean"] = metrics.get(
        "pragmatic_cue_capture_1_5_mean"
    )
    row[f"{prefix}_unsupported_inference_rate"] = metrics.get(
        "unsupported_inference_rate_mean"
    )
    row[f"{prefix}_unsafe_or_misleading_rate"] = metrics.get(
        "unsafe_or_misleading_rate_mean"
    )


def format_kappa_pairs(rows: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        kappa = row.get("kappa")
        if kappa is None:
            continue
        parts.append(
            f"{row.get('judge_a')} vs {row.get('judge_b')}: {float(kappa):.3f} (n={row.get('n')})"
        )
    return " ; ".join(parts)


def format_drop_self(rows: Sequence[Dict[str, Any]], source_model: str) -> str:
    for row in rows:
        if row.get("source_model") != source_model:
            continue
        aggregate = row.get("aggregate", {})
        all_mean = aggregate.get("all_judges_overall_success_rate_mean")
        drop_mean = aggregate.get("drop_self_overall_success_rate_mean")
        if all_mean is None or drop_mean is None:
            return ""
        dropped = ", ".join(row.get("dropped_judge_models", []))
        return f"drop {dropped}: {float(drop_mean):.4f} vs all {float(all_mean):.4f}"
    return ""


def load_generation_judge_rows() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    files = generation_judge_files()
    loaded_payloads = [(path, read_json(path)) for path in files]
    consensus_payloads = [
        (path, payload)
        for path, payload in loaded_payloads
        if payload.get("protocol") == "generation_judge_consensus_v1"
    ]
    if consensus_payloads:
        rows: List[Dict[str, Any]] = []
        input_files: List[Dict[str, Any]] = []
        pairwise_kappas: List[Dict[str, Any]] = []
        drop_self_ablation: List[Dict[str, Any]] = []
        for path, payload in consensus_payloads:
            input_files.append(
                {
                    "path": rel(path),
                    "protocol": payload.get("protocol"),
                    "judge_models": payload.get("judge_models", []),
                    "judgments": len(payload.get("judgments", [])),
                    "source_item_count": payload.get("summary", {}).get("source_item_count"),
                }
            )
            pairwise_kappas.extend(payload.get("pairwise_cohens_kappa", []))
            drop_self_ablation.extend(payload.get("drop_self_ablation", []))
            for summary in payload.get("model_summaries", []):
                source_model = summary.get("source_model") or "unknown"
                row: Dict[str, Any] = {
                    "model": summary.get("display_model") or model_display(source_model),
                    # Hidden field kept only so we can sort rows by the canonical
                    # MODEL_ORDER index of the raw source_model id, not the
                    # display name. CSV writers drop fields not in their column
                    # list, so this field never reaches the released CSV.
                    "_source_model": source_model,
                    "judge_models": " ; ".join(summary.get("judge_models", [])),
                    "source_files": " ; ".join(
                        sorted(rel(Path(path)) for path in summary.get("source_files", []))
                    ),
                    "notes": "3-judge ensemble; overall is mean ± sample SD across judge-level rates.",
                    "aggregate_pairwise_cohens_kappa_mean": summary.get("aggregate", {}).get(
                        "pairwise_cohens_kappa_mean"
                    ),
                    "pairwise_cohens_kappa": format_kappa_pairs(
                        summary.get("aggregate", {}).get("pairwise_cohens_kappa", [])
                    ),
                    "drop_self_ablation": format_drop_self(drop_self_ablation, source_model),
                }
                fill_generation_ensemble_metrics(row, prefix="core", metrics=summary.get("core"))
                fill_generation_ensemble_metrics(row, prefix="hard", metrics=summary.get("hard"))
                fill_generation_ensemble_metrics(
                    row, prefix="aggregate", metrics=summary.get("aggregate")
                )
                rows.append(row)
        rows.sort(key=lambda row: model_sort_key(row.get("_source_model") or row["model"]))
        return rows, input_files, {
            "protocol": "generation_judge_consensus_v1",
            "pairwise_cohens_kappa": pairwise_kappas,
            "drop_self_ablation": drop_self_ablation,
        }

    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rows": defaultdict(list),
            "judge_models": set(),
            "source_files": set(),
            "notes": [],
            "seen_source_ids": set(),
        }
    )
    input_files: List[Dict[str, Any]] = []

    for path, payload in loaded_payloads:
        input_files.append(
            {
                "path": rel(path),
                "judge_model": payload.get("judge_model"),
                "judge_provider": payload.get("judge_provider"),
                "judgments": len(payload.get("judgments", [])),
            }
        )
        for judgment in payload.get("judgments", []):
            split = generation_split_key(judgment.get("split_name"))
            if split is None:
                continue
            model = judgment.get("source_model") or "unknown"
            source_id = judgment.get("source_id")
            entry = by_model[model]
            if source_id and source_id in entry["seen_source_ids"]:
                continue
            if source_id:
                entry["seen_source_ids"].add(source_id)
            entry["rows"][split].append(judgment)
            entry["judge_models"].add(payload.get("judge_model") or "")
            entry["source_files"].add(rel(Path(judgment.get("source_file", path))))

    out: List[Dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        entry = by_model[model]
        core_metrics = generation_metrics(entry["rows"].get("core", []))
        hard_metrics = generation_metrics(entry["rows"].get("hard", []))
        aggregate_metrics = generation_metrics(
            list(entry["rows"].get("core", [])) + list(entry["rows"].get("hard", []))
        )
        row: Dict[str, Any] = {
            "model": model_display(model),
            "judge_models": " ; ".join(sorted(m for m in entry["judge_models"] if m)),
            "source_files": " ; ".join(sorted(entry["source_files"])),
            "notes": " ; ".join(entry["notes"]),
        }
        fill_generation_metrics(row, prefix="core", metrics=core_metrics)
        fill_generation_metrics(row, prefix="hard", metrics=hard_metrics)
        fill_generation_metrics(row, prefix="aggregate", metrics=aggregate_metrics)
        out.append(row)
    return out, input_files, {}


def validate_rows(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        for key in [
            "core_accuracy",
            "hard_accuracy",
            "hard_pragmatic_accuracy",
            "hard_direct_accuracy",
            "real_clean_accuracy",
            "aggregate_accuracy",
            "parse_failure_rate",
        ]:
            value = row.get(key)
            if value in {"", None}:
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{row['model']} {key} out of range: {value}")

    expected_counts = {
        "core_n": 3000,
        "hard_n": 469,
        "hard_pragmatic_n": 419,
        "hard_direct_n": 50,
        "real_clean_n": 64,
        "aggregate_n": 3533,
    }
    full_rows = [row for row in rows if row.get("aggregate_n") not in {"", None}]
    for row in full_rows:
        for key, expected in expected_counts.items():
            got = row.get(key)
            if got not in {"", None} and int(got) != expected:
                raise ValueError(f"{row['model']} {key}: expected {expected}, got {got}")


def table_tex(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
    r"\caption{Final sign-off-filtered recognition results. Scores are recomputed on PCP-Core-3000, PCP-Hard-469 (419 pragmatic items and 50 direct controls), PCP-Real-Clean-64, and the 3,533-item aggregate. Model rows are omitted unless item-level outputs are available for the final mask.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & \textbf{Core} & \textbf{Hard} & \textbf{Hard Prag.} & \textbf{Hard Direct} & \textbf{Real-Clean} & \textbf{Aggregate} "
        + LB,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']:28s} & {row['type']:8s} & "
            f"{pct(row.get('core_accuracy')):>5s} & "
            f"{pct(row.get('hard_accuracy')):>5s} & "
            f"{pct(row.get('hard_pragmatic_accuracy')):>5s} & "
            f"{pct(row.get('hard_direct_accuracy')):>5s} & "
            f"{pct(row.get('real_clean_accuracy')):>5s} & "
            f"{pct(row.get('aggregate_accuracy')):>5s} {LB}"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def classification_table_tex(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Final auxiliary phenomenon-classification results. Primary accuracy uses \texttt{phenomena\_primary} (or \texttt{DIRECT} for direct controls); Any reports whether the prediction matches any annotated phenomenon.}",
        r"\label{tab:classification-final-results}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Core P} & \textbf{Core Any} & \textbf{Hard P} & \textbf{Hard Any} & \textbf{Real P} & \textbf{Real Any} & \textbf{Agg. P} & \textbf{Agg. Any} "
        + LB,
        r"\midrule",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"{tex_escape(row['model'])} & "
                f"{pct(row.get('core_primary_accuracy')):>5s} & "
                f"{pct(row.get('core_any_phenomenon_accuracy')):>5s} & "
                f"{pct(row.get('hard_primary_accuracy')):>5s} & "
                f"{pct(row.get('hard_any_phenomenon_accuracy')):>5s} & "
                f"{pct(row.get('real_clean_primary_accuracy')):>5s} & "
                f"{pct(row.get('real_clean_any_phenomenon_accuracy')):>5s} & "
                f"{pct(row.get('aggregate_primary_accuracy')):>5s} & "
                f"{pct(row.get('aggregate_any_phenomenon_accuracy')):>5s} {LB}"
            )
    else:
        lines.append(r"\multicolumn{9}{c}{No complete final classification outputs available.} " + LB)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def generation_judge_table_tex(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{LLM-judge evaluation of free-text generation outputs on Core and Hard only. Overall is the mean $\pm$ sample standard deviation of binary success rates across judges when an ensemble is used; $\kappa$ is the aggregate mean pairwise Cohen's $\kappa$ over \texttt{overall\_success}. Intent, Action, and Cue are aggregate 1--5 means.}",
        r"\label{tab:generation-judge-results}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Core Overall} & \textbf{Hard Overall} & \textbf{Agg. Overall} & \textbf{$\kappa$} & \textbf{Intent} & \textbf{Action} & \textbf{Cue} & \textbf{Unsup.} & \textbf{Unsafe} "
        + LB,
        r"\midrule",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"{tex_escape(row['model'])} & "
                f"{pct_pm(row.get('core_overall_success_rate'), row.get('core_overall_success_rate_std')):>5s} & "
                f"{pct_pm(row.get('hard_overall_success_rate'), row.get('hard_overall_success_rate_std')):>5s} & "
                f"{pct_pm(row.get('aggregate_overall_success_rate'), row.get('aggregate_overall_success_rate_std')):>5s} & "
                f"{mean_1_5(row.get('aggregate_pairwise_cohens_kappa_mean')):>4s} & "
                f"{mean_1_5(row.get('aggregate_intent_match_mean')):>4s} & "
                f"{mean_1_5(row.get('aggregate_clinical_action_match_mean')):>4s} & "
                f"{mean_1_5(row.get('aggregate_pragmatic_cue_capture_mean')):>4s} & "
                f"{pct(row.get('aggregate_unsupported_inference_rate')):>5s} & "
                f"{pct(row.get('aggregate_unsafe_or_misleading_rate')):>5s} {LB}"
            )
    else:
        lines.append(r"\multicolumn{10}{c}{No generation judge outputs available.} " + LB)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def mean_1_5(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def main() -> None:
    core_tasks = load_compact_tasks(CORE_PATH)["recognition"]
    hard_tasks = load_compact_tasks(HARD_PATH)["recognition"]
    real_tasks = load_compact_tasks(REAL_PATH)["recognition"]

    expected_counts = {
        "core": len(core_tasks),
        "hard": len(hard_tasks),
        "real_clean": len(real_tasks),
    }
    rows = baseline_rows(core_tasks, hard_tasks, real_tasks)
    rows.extend(load_api_result_rows(expected_counts))

    validate_rows(rows)
    write_csv(SUMMARY_CSV, rows)
    TABLE_TEX.parent.mkdir(parents=True, exist_ok=True)
    TABLE_TEX.write_text(table_tex(rows), encoding="utf-8")

    classification_rows = load_api_classification_rows(expected_counts)
    write_csv(CLASSIFICATION_CSV, classification_rows, CLASSIFICATION_CSV_COLUMNS)
    CLASSIFICATION_TEX.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_TEX.write_text(
        classification_table_tex(classification_rows),
        encoding="utf-8",
    )
    classification_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "Final sign-off-filtered classification results",
        "metric": "phenomena_primary_accuracy",
        "auxiliary_metric": "any_phenomenon_accuracy",
        "policy": (
            "Primary classification accuracy uses phenomena_primary, with DIRECT "
            "as the gold label for direct controls. Any-phenomenon accuracy is "
            "reported as an auxiliary multi-label tolerance metric."
        ),
        "final_counts": {
            "Core": len(core_tasks),
            "Hard": len(hard_tasks),
            "Hard pragmatic": sum(1 for task in hard_tasks if task.get("variant") != "direct"),
            "Hard direct": sum(1 for task in hard_tasks if task.get("variant") == "direct"),
            "Real-Clean": len(real_tasks),
            "Aggregate": len(core_tasks) + len(hard_tasks) + len(real_tasks),
        },
        "raw_output_availability": missing_raw_outputs(),
        "rows": classification_rows,
        "output_files": {
            "csv": rel(CLASSIFICATION_CSV),
            "summary_json": rel(CLASSIFICATION_JSON),
            "paper_table": rel(CLASSIFICATION_TEX),
        },
    }
    write_json(CLASSIFICATION_JSON, classification_summary)

    generation_rows, generation_inputs, generation_protocol = load_generation_judge_rows()
    write_csv(GENERATION_JUDGE_CSV, generation_rows, GENERATION_JUDGE_CSV_COLUMNS)
    GENERATION_JUDGE_TEX.parent.mkdir(parents=True, exist_ok=True)
    GENERATION_JUDGE_TEX.write_text(
        generation_judge_table_tex(generation_rows),
        encoding="utf-8",
    )
    generation_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "Final generation judge results",
        "scope": "Core and Hard generation outputs only; Real-Clean is not a generation split.",
        "metrics": [
            "intent_match_1_5",
            "clinical_action_match_1_5",
            "pragmatic_cue_capture_1_5",
            "unsupported_inference",
            "unsafe_or_misleading",
            "overall_success",
        ],
        "input_judge_files": generation_inputs,
        "judge_protocol": generation_protocol,
        "rows": generation_rows,
        "output_files": {
            "csv": rel(GENERATION_JUDGE_CSV),
            "summary_json": rel(GENERATION_JUDGE_JSON),
            "paper_table": rel(GENERATION_JUDGE_TEX),
        },
    }
    write_json(GENERATION_JUDGE_JSON, generation_summary)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "Final sign-off-filtered model results",
        "metric": "recognition_accuracy",
        "policy": (
            "Rows are reported only when item-level outputs are available and can be "
            "filtered to final_inclusion_mask.csv. Aggregate-only or audit-only "
            "experiment records are not reused for final release reporting."
        ),
        "source_files": {
            "core": rel(CORE_PATH),
            "hard": rel(HARD_PATH),
            "real_clean": rel(REAL_PATH),
            "final_inclusion_mask": rel(MASK_PATH),
        },
        "final_counts": {
            "Core": len(core_tasks),
            "Hard": len(hard_tasks),
            "Hard pragmatic": sum(1 for task in hard_tasks if task.get("variant") != "direct"),
            "Hard direct": sum(1 for task in hard_tasks if task.get("variant") == "direct"),
            "Real-Clean": len(real_tasks),
            "Aggregate": len(core_tasks) + len(hard_tasks) + len(real_tasks),
        },
        "raw_output_availability": missing_raw_outputs(),
        "rows": rows,
        "output_files": {
            "csv": rel(SUMMARY_CSV),
            "summary_json": rel(SUMMARY_JSON),
            "paper_table": rel(TABLE_TEX),
            "classification_csv": rel(CLASSIFICATION_CSV),
            "classification_json": rel(CLASSIFICATION_JSON),
            "classification_tex": rel(CLASSIFICATION_TEX),
            "generation_judge_csv": rel(GENERATION_JUDGE_CSV),
            "generation_judge_json": rel(GENERATION_JUDGE_JSON),
            "generation_judge_tex": rel(GENERATION_JUDGE_TEX),
        },
        "excluded_reporting_sources": [
            "pre-signoff retained split counts",
            "pre-signoff total count",
            "summary-only API rows without final item-level outputs",
        ],
    }
    write_json(SUMMARY_JSON, summary)
    print(json.dumps({
        "summary_json": rel(SUMMARY_JSON),
        "csv": rel(SUMMARY_CSV),
        "paper_table": rel(TABLE_TEX),
        "final_counts": summary["final_counts"],
        "rows": [
            {
                "model": row["model"],
                "core": row.get("core_accuracy"),
                "hard": row.get("hard_accuracy"),
                "hard_pragmatic": row.get("hard_pragmatic_accuracy"),
                "hard_direct": row.get("hard_direct_accuracy"),
                "real_clean": row.get("real_clean_accuracy"),
                "aggregate": row.get("aggregate_accuracy"),
            }
            for row in rows
        ],
        "classification_outputs": {
            "csv": rel(CLASSIFICATION_CSV),
            "json": rel(CLASSIFICATION_JSON),
            "tex": rel(CLASSIFICATION_TEX),
            "rows": len(classification_rows),
        },
        "generation_judge_outputs": {
            "csv": rel(GENERATION_JUDGE_CSV),
            "json": rel(GENERATION_JUDGE_JSON),
            "tex": rel(GENERATION_JUDGE_TEX),
            "rows": len(generation_rows),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
