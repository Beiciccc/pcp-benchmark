#!/usr/bin/env python3
"""Judge free-text PCP generation outputs with an LLM rubric.

This script evaluates generation outputs qualitatively. It does not compute
exact-match accuracy. Each generation is scored against the reference pragmatic
intent and the source clinical utterance using structured JSON fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import stdev
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import write_json  # noqa: E402
from run_api_upper_bounds import (  # noqa: E402
    file_sha256,
    make_client,
    retry_complete,
    sha256_bytes,
    stable_json,
)


RUBRIC_FIELDS = [
    "intent_match_1_5",
    "clinical_action_match_1_5",
    "pragmatic_cue_capture_1_5",
    "unsupported_inference",
    "unsafe_or_misleading",
    "overall_success",
]

BINARY_FIELDS = ["unsupported_inference", "unsafe_or_misleading", "overall_success"]
SCORE_FIELDS = [
    "intent_match_1_5",
    "clinical_action_match_1_5",
    "pragmatic_cue_capture_1_5",
]
ENSEMBLE_PROTOCOL = "generation_judge_ensemble_v1"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"judgments": []}
    return read_json(path)


def input_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for value in args.input:
        paths.append(Path(value))
    for directory in args.input_dir:
        paths.extend(discover_pcp_full_files(Path(directory)))
    unique = deduplicate_input_paths(paths)
    if not unique:
        raise SystemExit("No input raw generation JSON files found.")
    return unique


def is_canonical_pcp_full(path: Path) -> bool:
    return path.name.endswith("_pcp_full.json")


def discover_pcp_full_files(directory: Path) -> List[Path]:
    """Find PCP full-result files, including Finder copy names.

    macOS Finder copies files as e.g. ``*_pcp_full 2.json``. Those copies do
    not match the canonical ``*_pcp_full.json`` glob, but some final raw files
    currently exist only under that copied name.
    """
    return sorted(
        path
        for path in directory.glob("*_pcp_full*.json")
        if "_pcp_full" in path.name
    )


def input_path_group_key(path: Path) -> tuple[str, str, str]:
    payload = read_json(path)
    split_name = payload.get("split_name") or payload.get("config", {}).get("split_name") or ""
    return (
        payload.get("provider") or "",
        payload.get("model") or "",
        split_name,
    )


def deduplicate_input_paths(paths: Sequence[Path]) -> List[Path]:
    """Deduplicate canonical/Finder-copy raw files for the same model split."""
    selected: Dict[tuple[str, str, str], Path] = {}
    # Keep symlink spellings intact so input metadata can use the canonical
    # symlink name instead of a Finder-copy target such as ``*_pcp_full 2.json``.
    explicit = {Path(path).absolute() for path in paths if Path(path).exists()}
    for path in sorted(explicit):
        key = input_path_group_key(path)
        previous = selected.get(key)
        if previous is None:
            selected[key] = path
            continue
        if is_canonical_pcp_full(path) and not is_canonical_pcp_full(previous):
            selected[key] = path
    return sorted(selected.values())


def source_key(path: Path, payload: Dict[str, Any], row: Dict[str, Any]) -> str:
    parts = [
        payload.get("provider") or "",
        payload.get("model") or "",
        payload.get("split_name") or payload.get("config", {}).get("split_name") or "",
        row.get("eval_id") or "",
        row.get("prompt_style") or "",
    ]
    return sha256_bytes(stable_json(parts).encode("utf-8"))


def generation_rows(
    paths: Sequence[Path],
    *,
    prompt_styles: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Collect generation rows across raw output files.

    If `prompt_styles` is provided, only rows whose `prompt_style` is in that
    set are kept. The filter is applied after the standard `error` skip so
    the source-id space stays disjoint per style.
    """
    style_filter = {s.strip() for s in prompt_styles or [] if s.strip()}
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        split_name = payload.get("split_name") or payload.get("config", {}).get("split_name")
        for row in payload.get("results", {}).get("generation", []):
            if row.get("error"):
                continue
            if style_filter and row.get("prompt_style") not in style_filter:
                continue
            rows.append(
                {
                    "source_id": source_key(path, payload, row),
                    "source_file": str(path),
                    "source_provider": payload.get("provider"),
                    "source_model": payload.get("model"),
                    "split_name": split_name,
                    **row,
                }
            )
    return rows


def judge_prompt(row: Dict[str, Any]) -> str:
    return (
        "You are judging a model's free-text answer for a clinical pragmatics "
        "benchmark. Evaluate whether the answer captures the speaker's intended "
        "communicative intent. Do not reward exact wording; reward faithful meaning.\n\n"
        f"Clinical context:\n{row.get('prompt', '')}\n\n"
        f"Reference intent:\n{row.get('generation_reference_intent', '')}\n\n"
        f"Model answer:\n{row.get('model_response', '')}\n\n"
        "Return JSON only with these fields:\n"
        "- intent_match_1_5: integer 1-5, where 5 means the main intent is captured.\n"
        "- clinical_action_match_1_5: integer 1-5, where 5 means the requested or implied clinical action/support matches.\n"
        "- pragmatic_cue_capture_1_5: integer 1-5, where 5 means indirectness, hedging, urgency, politeness, or implicature cues are captured when present.\n"
        "- unsupported_inference: yes/no, yes if the answer adds claims not supported by the utterance/context.\n"
        "- unsafe_or_misleading: yes/no, yes if the answer gives unsafe, misleading, stigmatizing, or clinically inappropriate interpretation/advice.\n"
        "- overall_success: yes/no, yes if the answer is good enough for qualitative success.\n"
        "- note: one brief sentence explaining the judgment.\n"
        "Every field is mandatory. Do not use null. Use \"no\" when a binary issue is absent.\n"
        "Do not wrap the JSON in Markdown or code fences.\n"
    )


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_yes_no(value: Any, field: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return "yes" if text == "true" else "no"
    if text not in {"yes", "no"}:
        raise ValueError(f"{field} must be yes/no, got {value!r}")
    return text


def normalize_score(value: Any, field: str) -> int:
    score = int(value)
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be 1-5, got {value!r}")
    return score


def parsed_value(parsed: Dict[str, Any], *field_names: str) -> Any:
    if not parsed:
        return None
    normalized = {str(key).strip(): value for key, value in parsed.items()}
    for field_name in field_names:
        if field_name in normalized:
            return normalized[field_name]
    return None


def normalize_judgment(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent_match_1_5": normalize_score(
            parsed_value(parsed, "intent_match_1_5"),
            "intent_match_1_5",
        ),
        "clinical_action_match_1_5": normalize_score(
            parsed_value(parsed, "clinical_action_match_1_5"),
            "clinical_action_match_1_5",
        ),
        "pragmatic_cue_capture_1_5": normalize_score(
            parsed_value(
                parsed,
                "pragmatic_cue_capture_1_5",
                "pragmaic_cue_capture_1_5",
                "pragma_cue_capture_1_5",
                "pragmatic_cue_capture_1_15",
            ),
            "pragmatic_cue_capture_1_5",
        ),
        "unsupported_inference": normalize_yes_no(
            parsed_value(parsed, "unsupported_inference"),
            "unsupported_inference",
        ),
        "unsafe_or_misleading": normalize_yes_no(
            parsed_value(parsed, "unsafe_or_misleading"),
            "unsafe_or_misleading",
        ),
        "overall_success": normalize_yes_no(
            parsed_value(parsed, "overall_success"),
            "overall_success",
        ),
        "note": str(parsed_value(parsed, "note") or "").strip(),
    }


def aggregate(judgments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in judgments if not row.get("error") and not row.get("parse_failed")]
    out: Dict[str, Any] = {
        "total_judgments": len(judgments),
        "valid_judgments": len(valid),
        "parse_failures": sum(1 for row in judgments if row.get("parse_failed")),
        "errors": sum(1 for row in judgments if row.get("error")),
    }
    for field in ["intent_match_1_5", "clinical_action_match_1_5", "pragmatic_cue_capture_1_5"]:
        out[f"{field}_mean"] = (
            sum(float(row[field]) for row in valid) / len(valid) if valid else None
        )
    for field in ["unsupported_inference", "unsafe_or_misleading", "overall_success"]:
        out[f"{field}_rate"] = (
            sum(1 for row in valid if row.get(field) == "yes") / len(valid) if valid else None
        )
    return out


def split_key(split_name: str | None) -> str | None:
    value = split_name or ""
    if "Core" in value:
        return "core"
    if "Hard" in value:
        return "hard"
    return None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: Sequence[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return stdev(values)


def yes_rate(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    valid = [row for row in rows if not row.get("error") and not row.get("parse_failed")]
    if not valid:
        return None
    return sum(1 for row in valid if row.get(field) == "yes") / len(valid)


def score_mean(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    valid = [row for row in rows if not row.get("error") and not row.get("parse_failed")]
    if not valid:
        return None
    return sum(float(row[field]) for row in valid) / len(valid)


def normalize_model_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def is_self_judge(source_model: str | None, judge_model: str | None) -> bool:
    source = normalize_model_name(source_model)
    judge = normalize_model_name(judge_model)
    aliases = {
        "gpt55": {"gpt55"},
        "claudeopus47": {"claudeopus47", "claude47opus"},
        "gemini31propreview": {"gemini31propreview"},
    }
    for source_alias, judge_aliases in aliases.items():
        if source_alias in source and any(alias in judge for alias in judge_aliases):
            return True
    return source == judge and bool(source)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Dict[str, Any]:
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equal-length label sequences.")
    n = len(labels_a)
    if not n:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    yes_no = ["yes", "no"]
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = 0.0
    for label in yes_no:
        pa = sum(a == label for a in labels_a) / n
        pb = sum(b == label for b in labels_b) / n
        expected += pa * pb
    if math.isclose(1.0 - expected, 0.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def judgment_rank(row: Dict[str, Any]) -> tuple[int, int]:
    usable = not row.get("error") and not row.get("parse_failed")
    return (1 if usable else 0, int(bool(row.get("raw_judge_response"))))


def load_judge_payloads(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if payload.get("protocol") == ENSEMBLE_PROTOCOL:
            continue
        judge_model = payload.get("judge_model")
        judge_provider = payload.get("judge_provider")
        by_source: Dict[str, Dict[str, Any]] = {}
        ranks: Dict[str, tuple[int, int]] = {}
        for row in payload.get("judgments", []):
            source_id = row.get("source_id")
            if not source_id:
                continue
            rank = judgment_rank(row)
            if source_id not in by_source or rank >= ranks[source_id]:
                by_source[source_id] = row
                ranks[source_id] = rank
        payloads.append(
            {
                "path": path,
                "judge_model": judge_model,
                "judge_provider": judge_provider,
                "judgments": list(by_source.values()),
            }
        )
    return payloads


def summarize_judge_rates(
    judgments: Sequence[Dict[str, Any]],
    *,
    allowed_judges: set[str] | None = None,
) -> Dict[str, Any]:
    by_judge: Dict[str, List[Dict[str, Any]]] = {}
    for row in judgments:
        judge_model = row.get("judge_model") or ""
        if allowed_judges is not None and judge_model not in allowed_judges:
            continue
        by_judge.setdefault(judge_model, []).append(row)
    rates = {
        judge: yes_rate(rows, "overall_success")
        for judge, rows in sorted(by_judge.items())
    }
    valid_rates = [float(rate) for rate in rates.values() if rate is not None]
    out: Dict[str, Any] = {
        "judge_count": len(valid_rates),
        "overall_success_rate_by_judge": rates,
        "overall_success_rate_mean": mean(valid_rates),
        "overall_success_rate_std": sample_std(valid_rates),
    }
    for field in SCORE_FIELDS:
        values = [
            score_mean(rows, field)
            for judge, rows in sorted(by_judge.items())
        ]
        valid_values = [float(value) for value in values if value is not None]
        out[f"{field}_mean"] = mean(valid_values)
        out[f"{field}_std"] = sample_std(valid_values)
    for field in ["unsupported_inference", "unsafe_or_misleading"]:
        values = [
            yes_rate(rows, field)
            for judge, rows in sorted(by_judge.items())
        ]
        valid_values = [float(value) for value in values if value is not None]
        out[f"{field}_rate_mean"] = mean(valid_values)
        out[f"{field}_rate_std"] = sample_std(valid_values)
    return out


def kappa_for_scope(
    judgments: Sequence[Dict[str, Any]],
    *,
    scope: str,
    source_model: str | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    by_judge_source: Dict[str, Dict[str, str]] = {}
    for row in judgments:
        if row.get("error") or row.get("parse_failed"):
            continue
        if row.get("overall_success") not in {"yes", "no"}:
            continue
        if source_model is not None and row.get("source_model") != source_model:
            continue
        if split is not None and split_key(row.get("split_name")) != split:
            continue
        judge = row.get("judge_model") or ""
        by_judge_source.setdefault(judge, {})[row["source_id"]] = row["overall_success"]
    out: List[Dict[str, Any]] = []
    for judge_a, judge_b in combinations(sorted(by_judge_source), 2):
        common = sorted(set(by_judge_source[judge_a]) & set(by_judge_source[judge_b]))
        stats = cohen_kappa(
            [by_judge_source[judge_a][source_id] for source_id in common],
            [by_judge_source[judge_b][source_id] for source_id in common],
        )
        out.append(
            {
                "scope": scope,
                "source_model": source_model,
                "split": split,
                "field": "overall_success",
                "judge_a": judge_a,
                "judge_b": judge_b,
                **stats,
            }
        )
    return out


def build_pairwise_kappas(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_models = sorted({row.get("source_model") for row in judgments if row.get("source_model")})
    out = kappa_for_scope(judgments, scope="all")
    for split in ["core", "hard"]:
        out.extend(kappa_for_scope(judgments, scope=f"split:{split}", split=split))
    for model in source_models:
        out.extend(kappa_for_scope(judgments, scope="source_model", source_model=model))
        for split in ["core", "hard"]:
            out.extend(
                kappa_for_scope(
                    judgments,
                    scope=f"source_model:{split}",
                    source_model=model,
                    split=split,
                )
            )
    return out


def kappa_mean(rows: Sequence[Dict[str, Any]]) -> float | None:
    values = [float(row["kappa"]) for row in rows if row.get("kappa") is not None]
    return mean(values)


def build_model_summaries(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in judgments:
        by_model.setdefault(row.get("source_model") or "unknown", []).append(row)
    kappas = build_pairwise_kappas(judgments)
    out: List[Dict[str, Any]] = []
    for model in sorted(by_model):
        rows = by_model[model]
        model_summary: Dict[str, Any] = {
            "source_model": model,
            "judge_models": sorted({row.get("judge_model") for row in rows if row.get("judge_model")}),
            "source_files": sorted({row.get("source_file") for row in rows if row.get("source_file")}),
        }
        for split in ["core", "hard"]:
            split_rows = [row for row in rows if split_key(row.get("split_name")) == split]
            model_summary[split] = summarize_judge_rates(split_rows)
            model_summary[split]["n_source_items"] = len(
                {row.get("source_id") for row in split_rows if row.get("source_id")}
            )
        model_summary["aggregate"] = summarize_judge_rates(rows)
        model_summary["aggregate"]["n_source_items"] = len(
            {row.get("source_id") for row in rows if row.get("source_id")}
        )
        model_kappas = [
            row for row in kappas
            if row.get("scope") == "source_model" and row.get("source_model") == model
        ]
        model_summary["aggregate"]["pairwise_cohens_kappa_mean"] = kappa_mean(model_kappas)
        model_summary["aggregate"]["pairwise_cohens_kappa"] = model_kappas
        out.append(model_summary)
    return out


def build_ensemble_items(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in judgments:
        by_source.setdefault(row["source_id"], []).append(row)
    out: List[Dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        valid = [row for row in rows if not row.get("error") and not row.get("parse_failed")]
        base = valid[0] if valid else rows[0]
        item: Dict[str, Any] = {
            "source_id": source_id,
            "source_model": base.get("source_model"),
            "source_file": base.get("source_file"),
            "split_name": base.get("split_name"),
            "eval_id": base.get("eval_id"),
            "probe_id": base.get("probe_id"),
            "prompt_style": base.get("prompt_style"),
            "judge_count": len(valid),
            "judge_models": sorted({row.get("judge_model") for row in valid if row.get("judge_model")}),
        }
        for field in SCORE_FIELDS:
            values = [float(row[field]) for row in valid if field in row]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_std"] = sample_std(values)
        for field in BINARY_FIELDS:
            values = [row.get(field) for row in valid if row.get(field) in {"yes", "no"}]
            yes_values = [1.0 if value == "yes" else 0.0 for value in values]
            item[f"{field}_yes_rate"] = mean(yes_values)
            item[f"{field}_majority"] = "yes" if mean(yes_values) is not None and mean(yes_values) >= 0.5 else "no"
        out.append(item)
    return out


def build_drop_self_ablation(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in sorted({row.get("source_model") for row in judgments if row.get("source_model")}):
        rows = [row for row in judgments if row.get("source_model") == model]
        self_judges = {
            row.get("judge_model") or ""
            for row in rows
            if is_self_judge(model, row.get("judge_model"))
        }
        if not self_judges:
            continue
        allowed = {
            row.get("judge_model") or ""
            for row in rows
            if (row.get("judge_model") or "") not in self_judges
        }
        ablation: Dict[str, Any] = {
            "source_model": model,
            "dropped_judge_models": sorted(self_judges),
        }
        for split_name, split_rows in {
            "core": [row for row in rows if split_key(row.get("split_name")) == "core"],
            "hard": [row for row in rows if split_key(row.get("split_name")) == "hard"],
            "aggregate": rows,
        }.items():
            all_stats = summarize_judge_rates(split_rows)
            drop_stats = summarize_judge_rates(split_rows, allowed_judges=allowed)
            ablation[split_name] = {
                "all_judges_overall_success_rate_mean": all_stats.get("overall_success_rate_mean"),
                "all_judges_overall_success_rate_std": all_stats.get("overall_success_rate_std"),
                "drop_self_overall_success_rate_mean": drop_stats.get("overall_success_rate_mean"),
                "drop_self_overall_success_rate_std": drop_stats.get("overall_success_rate_std"),
                "delta_drop_self_minus_all": (
                    drop_stats.get("overall_success_rate_mean") - all_stats.get("overall_success_rate_mean")
                    if drop_stats.get("overall_success_rate_mean") is not None
                    and all_stats.get("overall_success_rate_mean") is not None
                    else None
                ),
                "drop_self_judge_count": drop_stats.get("judge_count"),
            }
        out.append(ablation)
    return out


def write_ensemble_payload(output_path: Path, judge_paths: Sequence[Path]) -> None:
    payloads = load_judge_payloads(judge_paths)
    judgments: List[Dict[str, Any]] = []
    for payload in payloads:
        for row in payload["judgments"]:
            row = dict(row)
            row["judge_model"] = payload["judge_model"]
            row["judge_provider"] = payload["judge_provider"]
            row["judge_file"] = str(payload["path"])
            judgments.append(row)
    valid = [row for row in judgments if not row.get("error") and not row.get("parse_failed")]
    pairwise_kappas = build_pairwise_kappas(valid)
    model_summaries = build_model_summaries(valid)
    output = {
        "protocol": ENSEMBLE_PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_policy": (
            "Per-model overall_success is reported as the mean and sample standard "
            "deviation of judge-level success rates across judges. Pairwise Cohen's "
            "kappa is computed on binary overall_success labels over common source_ids."
        ),
        "input_judge_files": [
            {
                "path": str(payload["path"]),
                "judge_provider": payload["judge_provider"],
                "judge_model": payload["judge_model"],
                "judgments": len(payload["judgments"]),
            }
            for payload in payloads
        ],
        "judge_models": sorted({payload["judge_model"] for payload in payloads if payload["judge_model"]}),
        "rubric_fields": RUBRIC_FIELDS,
        "judgments": judgments,
        "ensemble_items": build_ensemble_items(valid),
        "model_summaries": model_summaries,
        "pairwise_cohens_kappa": pairwise_kappas,
        "pairwise_cohens_kappa_mean": kappa_mean([
            row for row in pairwise_kappas if row.get("scope") == "all"
        ]),
        "drop_self_ablation": build_drop_self_ablation(valid),
        "summary": {
            "total_judgments": len(judgments),
            "valid_judgments": len(valid),
            "judge_count": len({payload["judge_model"] for payload in payloads if payload["judge_model"]}),
            "source_item_count": len({row.get("source_id") for row in valid}),
            "source_model_count": len({row.get("source_model") for row in valid if row.get("source_model")}),
        },
    }
    write_json(output_path, output)


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = aggregate(payload.get("judgments", []))
    write_json(path, payload)


def merge_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple generation-judge JSON files into an ensemble protocol output."
    )
    parser.add_argument("--input", action="append", default=[], help="Single-judge JSON file.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory containing generation_judge_*.json single-judge files.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    paths: List[Path] = [Path(value) for value in args.input]
    for directory in args.input_dir:
        paths.extend(
            path for path in sorted(Path(directory).glob("generation_judge_*.json"))
            if path.name not in {"generation_judge_results.json", "generation_judge_ensemble_3judge.json"}
        )
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit("No single-judge generation_judge_*.json inputs found.")
    write_ensemble_payload(Path(args.output), paths)
    payload = read_json(Path(args.output))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved {args.output}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Raw *_pcp_full.json file.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory containing raw *_pcp_full.json files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        action="append",
        default=[],
        help=(
            "Restrict judging to generation rows with one of these prompt_style "
            "values (e.g. --prompt-style concise). Repeat the flag to accept "
            "multiple styles. Empty = all styles."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-rpm",
        type=float,
        default=0,
        help="Optional request-start cap for this judge process. 0 disables rate limiting.",
    )
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = input_paths(args)
    rows = generation_rows(paths, prompt_styles=args.prompt_style)
    if args.max_items:
        rows = rows[: args.max_items]

    style_note = (
        f" (prompt_style filter: {sorted(set(args.prompt_style))})"
        if args.prompt_style else ""
    )
    print(f"Generation rows to judge: {len(rows)}{style_note}")
    if args.dry_run:
        return

    output_path = Path(args.output)
    payload = load_existing(output_path)
    payload.update(
        {
            "judge_provider": args.provider,
            "judge_model": args.model,
            "rubric_fields": RUBRIC_FIELDS,
            "input_files": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in paths
            ],
        }
    )
    payload.setdefault("judgments", [])
    done = {row["source_id"] for row in payload["judgments"] if not row.get("error")}
    pending = [row for row in rows if row["source_id"] not in done]

    print(f"{len(payload['judgments'])} existing judgments, {len(pending)} pending")
    client = make_client(args.provider, args.model, request_timeout=args.request_timeout)

    min_start_interval = 60.0 / args.max_rpm if args.max_rpm and args.max_rpm > 0 else 0.0
    next_start_at = 0.0
    start_lock = threading.Lock()

    def throttle_start() -> None:
        nonlocal next_start_at
        if min_start_interval <= 0:
            return
        with start_lock:
            now = time.monotonic()
            wait_seconds = next_start_at - now
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            next_start_at = max(now, next_start_at) + min_start_interval

    def judge_one(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = judge_prompt(row)
        prompt_hash = sha256_bytes(prompt.encode("utf-8"))
        response = ""
        base = {
            "source_id": row["source_id"],
            "source_file": row["source_file"],
            "source_provider": row.get("source_provider"),
            "source_model": row.get("source_model"),
            "split_name": row.get("split_name"),
            "eval_id": row.get("eval_id"),
            "probe_id": row.get("probe_id"),
            "domain": row.get("domain"),
            "variant": row.get("variant"),
            "phenomena": row.get("phenomena", []),
            "prompt_style": row.get("prompt_style"),
            "generation_reference_intent": row.get("generation_reference_intent"),
            "model_response": row.get("model_response"),
            "generation_task_hash": row.get("task_hash"),
            "judge_prompt_hash": prompt_hash,
        }
        try:
            response, usage = retry_complete(
                client,
                prompt,
                max_tokens=args.max_tokens,
                retries=args.retries,
                sleep_seconds=args.sleep,
                before_attempt=throttle_start,
            )
            parsed = normalize_judgment(extract_json(response))
            return {
                **base,
                **parsed,
                "raw_judge_response": response,
                "parse_failed": False,
                "usage": usage,
            }
        except Exception as exc:  # noqa: BLE001 - preserve judge failure in output
            return {
                **base,
                "error": str(exc),
                "raw_judge_response": response,
                "parse_failed": True,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

    if args.concurrency <= 1:
        for idx, row in enumerate(pending, start=1):
            payload["judgments"].append(judge_one(row))
            if idx % args.save_every == 0 or idx == len(pending):
                write_payload(output_path, payload)
            if idx % args.log_every == 0 or idx == len(pending):
                print(f"  judged {idx}/{len(pending)}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(judge_one, row) for row in pending]
            for future in as_completed(futures):
                payload["judgments"].append(future.result())
                completed += 1
                if completed % args.save_every == 0 or completed == len(pending):
                    write_payload(output_path, payload)
                if completed % args.log_every == 0 or completed == len(pending):
                    print(f"  judged {completed}/{len(pending)}", flush=True)

    write_payload(output_path, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
