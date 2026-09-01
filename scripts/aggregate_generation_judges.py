#!/usr/bin/env python3
"""Aggregate multi-judge PCP generation judgments.

Inputs are single-judge outputs from ``judge_generation_outputs.py``. This
script aligns judgments by ``source_id``, writes per-row consensus decisions,
reports judge-level mean +/- standard deviation, computes pairwise Cohen's
kappa, and records drop-self ablations for self-judged model/judge pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import stdev
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "model_results"
DEFAULT_OUTPUT = OUT_DIR / "generation_judge_consensus_3judge.json"
DEFAULT_MAIN_TEX = ROOT / "paper" / "tables" / "generation_judge_results.tex"
DEFAULT_APPENDIX_TEX = ROOT / "paper" / "tables" / "generation_judge_appendix.tex"

PROTOCOL = "generation_judge_consensus_v1"
SCORE_FIELDS = [
    "intent_match_1_5",
    "clinical_action_match_1_5",
    "pragmatic_cue_capture_1_5",
]
BINARY_FIELDS = ["unsupported_inference", "unsafe_or_misleading", "overall_success"]
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
}
MODEL_ORDER = {model: idx for idx, model in enumerate(MODEL_DISPLAY)}
LB = r"\\"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def model_display(model: str | None) -> str:
    return MODEL_DISPLAY.get(model or "", model or "unknown")


def model_sort_key(model: str | None) -> tuple[int, str]:
    return (MODEL_ORDER.get(model or "", 99), model or "")


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


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * float(value):.1f}"


def pct_pm(value: float | None, std: float | None) -> str:
    if value is None:
        return "--"
    if std is None:
        return pct(value)
    return f"{100 * float(value):.1f}$\\pm${100 * float(std):.1f}"


def number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def split_key(split_name: str | None) -> str | None:
    value = split_name or ""
    if "Core" in value:
        return "core"
    if "Hard" in value:
        return "hard"
    return None


def normalize_model_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def is_self_judge(source_model: str | None, judge_model: str | None) -> bool:
    source = normalize_model_name(source_model)
    judge = normalize_model_name(judge_model)
    aliases = {
        "gpt55": {"gpt55"},
        "claudeopus47": {"claudeopus47", "claude47opus"},
        "gemini31propreview": {"gemini31propreview", "gemini31pro"},
    }
    for source_alias, judge_aliases in aliases.items():
        if source_alias in source and any(alias in judge for alias in judge_aliases):
            return True
    return source == judge and bool(source)


def judgment_rank(row: Dict[str, Any]) -> tuple[int, int]:
    usable = not row.get("error") and not row.get("parse_failed")
    return (1 if usable else 0, int(bool(row.get("raw_judge_response"))))


def discover_inputs(args: argparse.Namespace) -> List[Path]:
    paths = [Path(value) for value in args.input]
    for directory in args.input_dir:
        for path in sorted(Path(directory).glob("generation_judge_*.json")):
            if path.name in {
                "generation_judge_results.json",
                "generation_judge_consensus_3judge.json",
            }:
                continue
            payload = read_json(path)
            if payload.get("protocol") == PROTOCOL:
                continue
            paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("No single-judge generation_judge_*.json inputs found.")
    return unique


def load_single_judge_payloads(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if payload.get("protocol"):
            continue
        judge_model = payload.get("judge_model")
        judge_provider = payload.get("judge_provider")
        best_by_source: Dict[str, Dict[str, Any]] = {}
        ranks: Dict[str, tuple[int, int]] = {}
        for row in payload.get("judgments", []):
            source_id = row.get("source_id")
            if not source_id:
                continue
            rank = judgment_rank(row)
            if source_id not in best_by_source or rank >= ranks[source_id]:
                best_by_source[source_id] = row
                ranks[source_id] = rank
        payloads.append(
            {
                "path": path,
                "judge_model": judge_model,
                "judge_provider": judge_provider,
                "judgments": list(best_by_source.values()),
            }
        )
    return payloads


def flatten_judgments(payloads: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        for row in payload["judgments"]:
            row = dict(row)
            row["judge_model"] = payload.get("judge_model")
            row["judge_provider"] = payload.get("judge_provider")
            row["judge_file"] = rel(payload["path"])
            rows.append(row)
    return rows


def valid_judgment(row: Dict[str, Any]) -> bool:
    return not row.get("error") and not row.get("parse_failed")


def yes_rate(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    valid = [row for row in rows if valid_judgment(row) and row.get(field) in {"yes", "no"}]
    if not valid:
        return None
    return sum(1 for row in valid if row.get(field) == "yes") / len(valid)


def score_mean(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    valid = [row for row in rows if valid_judgment(row) and row.get(field) is not None]
    if not valid:
        return None
    return sum(float(row[field]) for row in valid) / len(valid)


def majority_label(labels: Sequence[str]) -> str | None:
    counts = {label: labels.count(label) for label in {"yes", "no"}}
    if counts["yes"] > counts["no"]:
        return "yes"
    if counts["no"] > counts["yes"]:
        return "no"
    return "unclear" if labels else None


def build_consensus_rows(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_source[row["source_id"]].append(row)

    out: List[Dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        valid = [row for row in rows if valid_judgment(row)]
        base = valid[0] if valid else rows[0]
        consensus: Dict[str, Any] = {
            "source_id": source_id,
            "source_model": base.get("source_model"),
            "source_file": base.get("source_file"),
            "split_name": base.get("split_name"),
            "eval_id": base.get("eval_id"),
            "probe_id": base.get("probe_id"),
            "domain": base.get("domain"),
            "variant": base.get("variant"),
            "phenomena": base.get("phenomena", []),
            "prompt_style": base.get("prompt_style"),
            "generation_reference_intent": base.get("generation_reference_intent"),
            "model_response": base.get("model_response"),
            "judge_count": len(valid),
            "judge_models": sorted({row.get("judge_model") for row in valid if row.get("judge_model")}),
            "missing_judges": sorted({row.get("judge_model") for row in rows if not valid_judgment(row)}),
        }
        for field in SCORE_FIELDS:
            values = [float(row[field]) for row in valid if row.get(field) is not None]
            consensus[f"{field}_mean"] = mean(values)
            consensus[f"{field}_std"] = sample_std(values)
        for field in BINARY_FIELDS:
            labels = [row[field] for row in valid if row.get(field) in {"yes", "no"}]
            yes_values = [1.0 if label == "yes" else 0.0 for label in labels]
            consensus[f"{field}_yes_rate"] = mean(yes_values)
            consensus[f"{field}_consensus"] = majority_label(labels)
        out.append(consensus)
    return out


def summarize_judge_rates(
    rows: Sequence[Dict[str, Any]],
    *,
    allowed_judges: set[str] | None = None,
) -> Dict[str, Any]:
    by_judge: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        judge_model = row.get("judge_model") or ""
        if allowed_judges is not None and judge_model not in allowed_judges:
            continue
        by_judge[judge_model].append(row)

    overall_rates = {
        judge: yes_rate(judge_rows, "overall_success")
        for judge, judge_rows in sorted(by_judge.items())
    }
    valid_rates = [float(rate) for rate in overall_rates.values() if rate is not None]
    out: Dict[str, Any] = {
        "judge_count": len(valid_rates),
        "n_source_items": len({row.get("source_id") for row in rows if row.get("source_id")}),
        "overall_success_rate_by_judge": overall_rates,
        "overall_success_rate_mean": mean(valid_rates),
        "overall_success_rate_std": sample_std(valid_rates),
    }
    for field in SCORE_FIELDS:
        values = [score_mean(judge_rows, field) for judge_rows in by_judge.values()]
        valid_values = [float(value) for value in values if value is not None]
        out[f"{field}_mean"] = mean(valid_values)
        out[f"{field}_std"] = sample_std(valid_values)
    for field in ["unsupported_inference", "unsafe_or_misleading"]:
        values = [yes_rate(judge_rows, field) for judge_rows in by_judge.values()]
        valid_values = [float(value) for value in values if value is not None]
        out[f"{field}_rate_mean"] = mean(valid_values)
        out[f"{field}_rate_std"] = sample_std(valid_values)
    return out


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Dict[str, Any]:
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equal-length inputs.")
    n = len(labels_a)
    if not n:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = 0.0
    for label in ["yes", "no"]:
        pa = sum(a == label for a in labels_a) / n
        pb = sum(b == label for b in labels_b) / n
        expected += pa * pb
    if math.isclose(1.0 - expected, 0.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {"n": n, "observed_agreement": observed, "expected_agreement": expected, "kappa": kappa}


def kappa_for_scope(
    judgments: Sequence[Dict[str, Any]],
    *,
    scope: str,
    field: str = "overall_success",
    source_model: str | None = None,
    split: str | None = None,
) -> List[Dict[str, Any]]:
    by_judge_source: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in judgments:
        if not valid_judgment(row):
            continue
        if row.get(field) not in {"yes", "no"}:
            continue
        if source_model is not None and row.get("source_model") != source_model:
            continue
        if split is not None and split_key(row.get("split_name")) != split:
            continue
        by_judge_source[row.get("judge_model") or ""][row["source_id"]] = row[field]

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
                "field": field,
                "judge_a": judge_a,
                "judge_b": judge_b,
                **stats,
            }
        )
    return out


def build_pairwise_kappas(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_models = sorted({row.get("source_model") for row in judgments if row.get("source_model")})
    rows: List[Dict[str, Any]] = []
    for field in BINARY_FIELDS:
        rows.extend(kappa_for_scope(judgments, scope="all", field=field))
        for split in ["core", "hard"]:
            rows.extend(kappa_for_scope(judgments, scope=f"split:{split}", field=field, split=split))
        for model in source_models:
            rows.extend(kappa_for_scope(judgments, scope="source_model", field=field, source_model=model))
            for split in ["core", "hard"]:
                rows.extend(
                    kappa_for_scope(
                        judgments,
                        scope=f"source_model:{split}",
                        field=field,
                        source_model=model,
                        split=split,
                    )
                )
    return rows


def kappa_mean(rows: Sequence[Dict[str, Any]], *, field: str = "overall_success") -> float | None:
    values = [
        float(row["kappa"])
        for row in rows
        if row.get("field") == field and row.get("kappa") is not None
    ]
    return mean(values)


def build_model_summaries(
    judgments: Sequence[Dict[str, Any]],
    pairwise_kappas: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_model[row.get("source_model") or "unknown"].append(row)

    summaries: List[Dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        rows = by_model[model]
        model_summary: Dict[str, Any] = {
            "source_model": model,
            "display_model": model_display(model),
            "judge_models": sorted({row.get("judge_model") for row in rows if row.get("judge_model")}),
            "source_files": sorted({rel(row.get("source_file")) for row in rows if row.get("source_file")}),
        }
        for split in ["core", "hard"]:
            split_rows = [row for row in rows if split_key(row.get("split_name")) == split]
            model_summary[split] = summarize_judge_rates(split_rows)
        model_summary["aggregate"] = summarize_judge_rates(rows)
        model_kappas = [
            row
            for row in pairwise_kappas
            if row.get("scope") == "source_model"
            and row.get("source_model") == model
            and row.get("field") == "overall_success"
        ]
        model_summary["aggregate"]["pairwise_cohens_kappa_mean"] = kappa_mean(model_kappas)
        model_summary["aggregate"]["pairwise_cohens_kappa"] = model_kappas
        summaries.append(model_summary)
    return summaries


def build_drop_self_ablation(judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    models = sorted({row.get("source_model") for row in judgments if row.get("source_model")})
    for model in models:
        rows = [row for row in judgments if row.get("source_model") == model]
        self_judges = {
            row.get("judge_model") or ""
            for row in rows
            if is_self_judge(model, row.get("judge_model"))
        }
        if not self_judges:
            continue
        allowed_judges = {
            row.get("judge_model") or ""
            for row in rows
            if (row.get("judge_model") or "") not in self_judges
        }
        entry: Dict[str, Any] = {
            "source_model": model,
            "display_model": model_display(model),
            "dropped_judge_models": sorted(self_judges),
        }
        for split_name, split_rows in {
            "core": [row for row in rows if split_key(row.get("split_name")) == "core"],
            "hard": [row for row in rows if split_key(row.get("split_name")) == "hard"],
            "aggregate": rows,
        }.items():
            all_stats = summarize_judge_rates(split_rows)
            drop_stats = summarize_judge_rates(split_rows, allowed_judges=allowed_judges)
            all_mean = all_stats.get("overall_success_rate_mean")
            drop_mean = drop_stats.get("overall_success_rate_mean")
            entry[split_name] = {
                "all_judges_overall_success_rate_mean": all_mean,
                "all_judges_overall_success_rate_std": all_stats.get("overall_success_rate_std"),
                "drop_self_overall_success_rate_mean": drop_mean,
                "drop_self_overall_success_rate_std": drop_stats.get("overall_success_rate_std"),
                "delta_drop_self_minus_all": (
                    drop_mean - all_mean
                    if drop_mean is not None and all_mean is not None
                    else None
                ),
                "drop_self_judge_count": drop_stats.get("judge_count"),
            }
        out.append(entry)
    return out


def format_kappa_pairs(rows: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        if row.get("kappa") is None:
            continue
        parts.append(
            f"{row.get('judge_a')} vs {row.get('judge_b')}: "
            f"{float(row['kappa']):.3f} (n={row.get('n')})"
        )
    return " ; ".join(parts)


def main_table_tex(model_summaries: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Three-judge LLM evaluation of free-text generation outputs on Core and Hard. Overall is the mean $\pm$ sample standard deviation of judge-level binary success rates; $\kappa$ is aggregate mean pairwise Cohen's $\kappa$ for \texttt{overall\_success}.}",
        r"\label{tab:generation-judge-results}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Core Overall} & \textbf{Hard Overall} & \textbf{Agg. Overall} & \textbf{$\kappa$} & \textbf{Intent} & \textbf{Action} & \textbf{Cue} & \textbf{Unsup.} & \textbf{Unsafe} "
        + LB,
        r"\midrule",
    ]
    if model_summaries:
        for summary in model_summaries:
            aggregate = summary.get("aggregate", {})
            lines.append(
                f"{tex_escape(summary.get('display_model'))} & "
                f"{pct_pm(summary.get('core', {}).get('overall_success_rate_mean'), summary.get('core', {}).get('overall_success_rate_std'))} & "
                f"{pct_pm(summary.get('hard', {}).get('overall_success_rate_mean'), summary.get('hard', {}).get('overall_success_rate_std'))} & "
                f"{pct_pm(aggregate.get('overall_success_rate_mean'), aggregate.get('overall_success_rate_std'))} & "
                f"{number(aggregate.get('pairwise_cohens_kappa_mean'))} & "
                f"{number(aggregate.get('intent_match_1_5_mean'))} & "
                f"{number(aggregate.get('clinical_action_match_1_5_mean'))} & "
                f"{number(aggregate.get('pragmatic_cue_capture_1_5_mean'))} & "
                f"{pct(aggregate.get('unsupported_inference_rate_mean'))} & "
                f"{pct(aggregate.get('unsafe_or_misleading_rate_mean'))} {LB}"
            )
    else:
        lines.append(r"\multicolumn{10}{c}{No generation judge consensus available.} " + LB)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def appendix_table_tex(
    pairwise_kappas: Sequence[Dict[str, Any]],
    drop_self: Sequence[Dict[str, Any]],
) -> str:
    overall = [
        row
        for row in pairwise_kappas
        if row.get("scope") == "all" and row.get("field") == "overall_success"
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Generation-judge agreement and drop-self ablations. Cohen's $\kappa$ is computed on binary \texttt{overall\_success}. Drop-self reports aggregate success after removing a judge that matches the evaluated model family.}",
        r"\label{tab:generation-judge-appendix}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Section} & \textbf{Comparison} & \textbf{n} & \textbf{$\kappa$ / All} & \textbf{Drop-self} "
        + LB,
        r"\midrule",
    ]
    for row in overall:
        kappa = "--" if row.get("kappa") is None else f"{float(row['kappa']):.3f}"
        lines.append(
            f"Pairwise & {tex_escape(model_display(row.get('judge_a')))} vs "
            f"{tex_escape(model_display(row.get('judge_b')))} & "
            f"{row.get('n')} & {kappa} & -- {LB}"
        )
    for row in drop_self:
        aggregate = row.get("aggregate", {})
        all_rate = aggregate.get("all_judges_overall_success_rate_mean")
        drop_rate = aggregate.get("drop_self_overall_success_rate_mean")
        lines.append(
            f"Drop-self & {tex_escape(row.get('display_model'))} (drop self) & "
            f"-- & {pct(all_rate)} & {pct(drop_rate)} {LB}"
        )
    if not overall and not drop_self:
        lines.append(r"\multicolumn{5}{c}{No agreement data available.} " + LB)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_payload(input_paths: Sequence[Path]) -> Dict[str, Any]:
    judge_payloads = load_single_judge_payloads(input_paths)
    judgments = flatten_judgments(judge_payloads)
    valid = [row for row in judgments if valid_judgment(row)]
    consensus_rows = build_consensus_rows(judgments)
    pairwise_kappas = build_pairwise_kappas(valid)
    model_summaries = build_model_summaries(valid, pairwise_kappas)
    drop_self = build_drop_self_ablation(valid)
    return {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_policy": (
            "Main generation results use per-model judge-level overall_success "
            "rates summarized as mean +/- sample standard deviation across judges. "
            "Consensus rows store majority binary labels and mean score fields. "
            "Cohen's kappa is pairwise over common source_ids."
        ),
        "input_judge_files": [
            {
                "path": rel(payload["path"]),
                "judge_provider": payload.get("judge_provider"),
                "judge_model": payload.get("judge_model"),
                "judgments": len(payload.get("judgments", [])),
            }
            for payload in judge_payloads
        ],
        "judge_models": sorted({payload.get("judge_model") for payload in judge_payloads if payload.get("judge_model")}),
        "rubric_fields": SCORE_FIELDS + BINARY_FIELDS,
        "judgments": judgments,
        "consensus_rows": consensus_rows,
        "model_summaries": model_summaries,
        "pairwise_cohens_kappa": pairwise_kappas,
        "drop_self_ablation": drop_self,
        "summary": {
            "judge_count": len({payload.get("judge_model") for payload in judge_payloads if payload.get("judge_model")}),
            "total_judgments": len(judgments),
            "valid_judgments": len(valid),
            "source_item_count": len({row.get("source_id") for row in judgments}),
            "valid_source_item_count": len({row.get("source_id") for row in valid}),
            "source_model_count": len({row.get("source_model") for row in valid if row.get("source_model")}),
            "overall_success_pairwise_kappa_mean": kappa_mean(
                [row for row in pairwise_kappas if row.get("scope") == "all"],
                field="overall_success",
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Single-judge JSON file.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory containing single-judge generation_judge_*.json files.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--main-tex", default=str(DEFAULT_MAIN_TEX))
    parser.add_argument("--appendix-tex", default=str(DEFAULT_APPENDIX_TEX))
    args = parser.parse_args()

    input_paths = discover_inputs(args)
    payload = build_payload(input_paths)
    output_path = Path(args.output)
    write_json(output_path, payload)
    write_text(Path(args.main_tex), main_table_tex(payload["model_summaries"]))
    write_text(
        Path(args.appendix_tex),
        appendix_table_tex(payload["pairwise_cohens_kappa"], payload["drop_self_ablation"]),
    )
    print(
        json.dumps(
            {
                "output": rel(output_path),
                "main_tex": rel(Path(args.main_tex)),
                "appendix_tex": rel(Path(args.appendix_tex)),
                **payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
