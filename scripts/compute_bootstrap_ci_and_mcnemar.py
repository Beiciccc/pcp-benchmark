#!/usr/bin/env python3
"""Bootstrap 95% CI + pairwise McNemar tests over the final recognition mask.

For every (model, split) cell of the final recognition results, this script
reads the per-probe majority-voted correctness vector and computes:

* Accuracy point estimate.
* Percentile bootstrap 95% CI (1000 resamples by default, seeded). This is a
  plain percentile interval over the per-probe correctness vector, not a
  bias-corrected (BCa) interval.

For every (model_a, model_b, split) pair this script computes:

* McNemar exact test on the b/c contingency table of per-probe (correct_a,
  correct_b) outcomes, with the small-sample binomial form when b + c <= 25
  and a chi-square approximation otherwise.
* The corresponding paired accuracy delta with bootstrap 95% CI.

Outputs:

* experiments/model_results/bootstrap_ci.json
* experiments/model_results/pairwise_mcnemar.json
* paper/tables/significance_results.tex   (appendix-ready summary table)

The script reads item-level outputs directly from
``experiments/api_*_final/*_pcp_full.json`` so it captures every model that
``summarize_final_model_results.py`` already reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pcp_eval_utils import majority_vote_by_probe  # noqa: E402


PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENT_DIR = ROOT / "experiments"
OUTPUT_MODEL_DIR = EXPERIMENT_DIR / "model_results"
PAPER_TABLES_DIR = ROOT / "paper" / "tables"
MASK_PATH = ROOT / "data" / "validation" / "final_inclusion_mask.csv"

API_RESULT_DIRS = {
    "core": EXPERIMENT_DIR / "api_core_3000_final",
    "hard": EXPERIMENT_DIR / "api_hard_469_final",
    "real_clean": EXPERIMENT_DIR / "api_real_clean_64_final",
}

SPLIT_TO_FINAL_PROBE_FILE = {
    "core": PROCESSED_DIR / "pcp_core_3000_final.json",
    "hard": PROCESSED_DIR / "pcp_hard_469_signoff_final.json",
    "real_clean": PROCESSED_DIR / "pcp_real_clean_64_signoff_final.json",
}

# Reuse the canonical display map from summarize_final_model_results.
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_final_model_results import (  # noqa: E402
    MODEL_DISPLAY,
    MODEL_ORDER,
    dedupe_api_results,
    model_display,
    model_sort_key,
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_seed(base_seed: int, *parts: str) -> int:
    """Derive a reproducible per-cell seed without Python's salted hash()."""
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def load_inclusion_mask() -> Dict[str, set]:
    mask: Dict[str, set] = {"core": set(), "hard": set(), "real_clean": set()}
    if not MASK_PATH.exists():
        return mask
    with MASK_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("included_in_main_eval", "").strip().lower() != "true":
                continue
            split = row.get("split", "").strip().lower()
            split_key = {
                "core": "core",
                "hard": "hard",
                "real-clean": "real_clean",
                "real_clean": "real_clean",
            }.get(split)
            if split_key:
                mask[split_key].add(row.get("probe_id", ""))
    return mask


def per_probe_correctness(api_rows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """Return canonical {probe_id: correct} using the main-table scoring path.

    This intentionally mirrors ``summarize_final_model_results.py``:
    first keep the best reusable row per ``eval_id`` in deterministic order,
    then majority-vote at probe level. Parse-failed probes are retained as
    incorrect if no valid shuffled run remains, so CI/McNemar denominators
    match the released recognition table.
    """
    deduped = dedupe_api_results(api_rows, "recognition")
    if not deduped:
        return {}
    majority = majority_vote_by_probe(deduped)
    return {r["probe_id"]: bool(r.get("correct")) for r in majority if "probe_id" in r}


def load_model_results_for_split(split: str, mask: set) -> Dict[str, Dict[str, bool]]:
    """Return {model_id_raw: {probe_id: correct}} restricted to mask."""
    out: Dict[str, Dict[str, bool]] = {}
    api_dir = API_RESULT_DIRS[split]
    if not api_dir.exists():
        return out
    for path in sorted(api_dir.glob("*_pcp_full.json")):
        payload = read_json(path)
        model = payload.get("model") or path.stem
        rows = payload.get("results", {}).get("recognition") or []
        if not rows:
            continue
        per_probe = per_probe_correctness(rows)
        if mask:
            per_probe = {pid: ok for pid, ok in per_probe.items() if pid in mask}
        if per_probe:
            out[model] = per_probe
    return out


def bootstrap_ci(correctness: Sequence[bool], n_resamples: int, seed: int) -> Tuple[float, float, float]:
    """Return (mean, ci_low_2.5%, ci_high_97.5%) for a 0/1 vector."""
    n = len(correctness)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = [1.0 if c else 0.0 for c in correctness]
    point = sum(arr) / n
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            s += arr[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    low_idx = max(0, int(round(0.025 * (n_resamples - 1))))
    high_idx = min(n_resamples - 1, int(round(0.975 * (n_resamples - 1))))
    return point, samples[low_idx], samples[high_idx]


def mcnemar_test(b: int, c: int) -> Dict[str, Any]:
    """Two-sided McNemar test on discordant counts b (a-only correct) and c (b-only).

    Uses exact binomial test for small b+c, chi-square continuity-corrected
    approximation for larger samples.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "trivial"}
    if n <= 25:
        # Exact two-sided binomial test with p=0.5.
        from math import comb
        # P(X <= min(b,c)) under Binomial(n, 0.5)
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(k + 1))
        p = min(1.0, (tail / (2 ** n)) * 2)
        method = "exact_binomial"
    else:
        # Continuity-corrected chi-square approximation.
        chi2 = ((abs(b - c) - 1) ** 2) / n
        # Survival of chi-square with df=1: P(chi2 >= x) = erfc(sqrt(x/2))
        p = math.erfc(math.sqrt(chi2 / 2.0))
        method = "chi2_continuity"
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "method": method}


def paired_accuracy_delta(a: Dict[str, bool], b: Dict[str, bool], n_resamples: int, seed: int) -> Dict[str, Any]:
    common = sorted(set(a) & set(b))
    if not common:
        return {"common": 0, "delta_point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    diff = [1.0 if a[pid] else 0.0 for pid in common]
    diff_b = [1.0 if b[pid] else 0.0 for pid in common]
    paired = [diff[i] - diff_b[i] for i in range(len(common))]
    point = sum(paired) / len(paired)
    rng = random.Random(seed)
    samples = []
    n = len(paired)
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            s += paired[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    return {
        "common": n,
        "delta_point": point,
        "ci_low": samples[max(0, int(round(0.025 * (n_resamples - 1))))],
        "ci_high": samples[min(n_resamples - 1, int(round(0.975 * (n_resamples - 1))))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--include-baselines", action="store_true",
                        help="Compute CI for Random and Always-A as sanity checks.")
    args = parser.parse_args()

    mask = load_inclusion_mask()

    # Per (model, split): item-level correctness vector under mask.
    per_split_per_model: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for split in API_RESULT_DIRS:
        per_split_per_model[split] = load_model_results_for_split(split, mask[split])

    all_models = sorted(
        {m for split in per_split_per_model.values() for m in split},
        key=model_sort_key,
    )

    print(f"[bootstrap_ci] models discovered: {len(all_models)}")
    for m in all_models:
        present = [s for s in API_RESULT_DIRS if m in per_split_per_model[s]]
        print(f"   {m:55s} → splits: {present}")

    # Bootstrap CI per (model, split, slice).
    ci_rows: List[Dict[str, Any]] = []
    for model in all_models:
        for split in ["core", "hard", "real_clean"]:
            data = per_split_per_model[split].get(model)
            if not data:
                continue
            # full slice
            corr = list(data.values())
            point, low, high = bootstrap_ci(corr, args.n_resamples, stable_seed(args.seed, model, split))
            ci_rows.append({
                "model": model,
                "display_model": model_display(model),
                "split": split,
                "slice": "all",
                "n": len(corr),
                "accuracy": round(point, 6),
                "ci_low": round(low, 6),
                "ci_high": round(high, 6),
            })
            # Hard pragmatic vs direct slices
            if split == "hard":
                probe_meta = read_json(SPLIT_TO_FINAL_PROBE_FILE[split])
                variant_by_probe = {p["id"]: p.get("variant") for p in probe_meta.get("probes", [])}
                prag = [data[pid] for pid, v in variant_by_probe.items() if pid in data and v != "direct"]
                direct = [data[pid] for pid, v in variant_by_probe.items() if pid in data and v == "direct"]
                for slice_name, slice_vec in [("pragmatic", prag), ("direct", direct)]:
                    if slice_vec:
                        p, lo, hi = bootstrap_ci(
                            slice_vec,
                            args.n_resamples,
                            stable_seed(args.seed, model, split, slice_name),
                        )
                        ci_rows.append({
                            "model": model,
                            "display_model": model_display(model),
                            "split": "hard",
                            "slice": slice_name,
                            "n": len(slice_vec),
                            "accuracy": round(p, 6),
                            "ci_low": round(lo, 6),
                            "ci_high": round(hi, 6),
                        })

    ci_path = OUTPUT_MODEL_DIR / "bootstrap_ci.json"
    write_json(ci_path, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "rows": ci_rows,
    })
    print(f"[bootstrap_ci] wrote {ci_path}  ({len(ci_rows)} rows)")

    # McNemar pairwise tests per (model_a, model_b, split).
    mcnemar_rows: List[Dict[str, Any]] = []
    for split in ["hard", "real_clean"]:
        models_in_split = sorted(
            (m for m in per_split_per_model[split]),
            key=model_sort_key,
        )
        for i, mi in enumerate(models_in_split):
            for mj in models_in_split[i + 1:]:
                a = per_split_per_model[split][mi]
                b = per_split_per_model[split][mj]
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                bb = sum(1 for pid in common if a[pid] and not b[pid])
                cc = sum(1 for pid in common if b[pid] and not a[pid])
                test = mcnemar_test(bb, cc)
                delta = paired_accuracy_delta(
                    a,
                    b,
                    args.n_resamples,
                    stable_seed(args.seed, mi, mj, split),
                )
                mcnemar_rows.append({
                    "split": split,
                    "model_a": mi,
                    "display_a": model_display(mi),
                    "model_b": mj,
                    "display_b": model_display(mj),
                    "n_paired": len(common),
                    "a_correct_only": bb,
                    "b_correct_only": cc,
                    "n_both_correct": sum(1 for pid in common if a[pid] and b[pid]),
                    "n_neither_correct": sum(1 for pid in common if not a[pid] and not b[pid]),
                    **test,
                    "delta_accuracy_a_minus_b": delta["delta_point"],
                    "delta_ci_low": delta["ci_low"],
                    "delta_ci_high": delta["ci_high"],
                })

    mc_path = OUTPUT_MODEL_DIR / "pairwise_mcnemar.json"
    write_json(mc_path, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_resamples_for_delta_ci": args.n_resamples,
        "seed": args.seed,
        "rows": mcnemar_rows,
    })
    print(f"[mcnemar] wrote {mc_path}  ({len(mcnemar_rows)} pairs)")

    # Compact appendix tex.
    write_significance_tex(ci_rows, mcnemar_rows)


def write_significance_tex(ci_rows: List[Dict[str, Any]], mcnemar_rows: List[Dict[str, Any]]) -> None:
    LB = r"\\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Bootstrap 95\% CI on recognition accuracy and pairwise McNemar tests (Hard and Real-Clean). Bootstrap uses 1000 paired resamples; McNemar is exact binomial for n$_{\text{discordant}}\leq25$ and continuity-corrected $\chi^2$ otherwise.}",
        r"\label{tab:significance-results}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"\textbf{Section} & \textbf{Model / Pair} & \textbf{Split / Slice} & \textbf{n} & \textbf{Acc.} & \textbf{CI low} & \textbf{CI high} " + LB,
        r"\midrule",
    ]
    for row in ci_rows:
        slice_label = row["slice"] if row["slice"] != "all" else ""
        lines.append(
            f"CI & {row['display_model']} & {row['split']}{(' / ' + slice_label) if slice_label else ''} & "
            f"{row['n']} & {100*row['accuracy']:.1f} & {100*row['ci_low']:.1f} & {100*row['ci_high']:.1f} {LB}"
        )
    lines.append(r"\midrule")
    lines.append(r"\textbf{Section} & \textbf{Pair} & \textbf{Split} & \textbf{n} & \textbf{$\Delta$Acc} & \textbf{p-value} & \textbf{method} " + LB)
    for row in sorted(mcnemar_rows, key=lambda r: (r["split"], r.get("p_value", 1.0))):
        delta = 100 * row["delta_accuracy_a_minus_b"]
        pv = row["p_value"]
        pv_label = "<1e-6" if pv < 1e-6 else f"{pv:.2e}" if pv < 0.001 else f"{pv:.3f}"
        lines.append(
            f"McNemar & {row['display_a']} vs {row['display_b']} & {row['split']} & "
            f"{row['n_paired']} & {delta:+.1f} & {pv_label} & {row['method'].replace('_', '-')} {LB}"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = PAPER_TABLES_DIR / "significance_results.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tex] wrote {out_path}")


if __name__ == "__main__":
    main()
