#!/usr/bin/env python3
"""Within-block (scenario x context) pragmatic contrast on PCP-Core with a
cluster bootstrap 95% CI.

Each of the 300 PCP-Core scenario--context blocks contains one ``direct`` probe
and nine pragmatic variants of the same content. For every model and block we
compute the within-block contrast ``direct_acc - mean(pragmatic_acc)`` (here
``direct_acc`` is a single 0/1 probe), then summarise across the 11 evaluated
systems. Uncertainty is a block-level (cluster) bootstrap that resamples whole
blocks, so it respects the block structure rather than treating probes as
independent.

This reproduces the figures quoted in Section 5.2 (per-model means whose
*median* is +0.74 pp) and replaces the previously reported pooled mean with the
exact recomputed value plus its CI.

Outputs:
* experiments/model_results/within_block_contrast.json

Run: python3 scripts/compute_within_block_ci.py
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pcp_eval_utils import majority_vote_by_probe  # noqa: E402
from summarize_final_model_results import (  # noqa: E402
    dedupe_api_results,
    model_display,
    model_sort_key,
)

CORE_DIR = ROOT / "experiments" / "api_core_3000_final"
CORE_META = ROOT / "data" / "processed" / "pcp_core_3000_final.json"
MASK_PATH = ROOT / "data" / "validation" / "final_inclusion_mask.csv"
OUT_PATH = ROOT / "experiments" / "model_results" / "within_block_contrast.json"
N_RESAMPLES = 1000
SEED = 20260516


def load_core_mask() -> set:
    mask: set = set()
    if not MASK_PATH.exists():
        return mask
    with MASK_PATH.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("included_in_main_eval", "").strip().lower() != "true":
                continue
            if row.get("split", "").strip().lower() == "core":
                mask.add(row.get("probe_id", ""))
    return mask


def load_block_variant() -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    meta = json.loads(CORE_META.read_text(encoding="utf-8"))
    block_of: Dict[str, Tuple[str, str]] = {}
    variant_of: Dict[str, str] = {}
    for p in meta["probes"]:
        block_of[p["id"]] = (p["scenario_type"], p["context"])
        variant_of[p["id"]] = p["variant"]
    return block_of, variant_of


def load_per_model(mask: set) -> Dict[str, Dict[str, bool]]:
    per_model: Dict[str, Dict[str, bool]] = {}
    for path in sorted(CORE_DIR.glob("*_pcp_full.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model") or path.stem
        rows = payload.get("results", {}).get("recognition") or []
        if not rows:
            continue
        maj = majority_vote_by_probe(dedupe_api_results(rows, "recognition"))
        corr = {r["probe_id"]: bool(r.get("correct")) for r in maj if "probe_id" in r}
        if mask:
            corr = {k: v for k, v in corr.items() if k in mask}
        if corr:
            per_model[model] = corr
    return per_model


def contrast_per_model_block(
    per_model: Dict[str, Dict[str, bool]],
    block_of: Dict[str, Tuple[str, str]],
    variant_of: Dict[str, str],
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """{model: {block: direct_acc - mean(pragmatic_acc)}}."""
    out: Dict[str, Dict[Tuple[str, str], float]] = {}
    for m, corr in per_model.items():
        bl = defaultdict(lambda: {"d": [], "p": []})
        for pid, ok in corr.items():
            b = block_of.get(pid)
            v = variant_of.get(pid)
            if b is None or v is None:
                continue
            (bl[b]["d"] if v == "direct" else bl[b]["p"]).append(1.0 if ok else 0.0)
        cb: Dict[Tuple[str, str], float] = {}
        for b, x in bl.items():
            if x["d"] and x["p"]:
                cb[b] = (sum(x["d"]) / len(x["d"])) - (sum(x["p"]) / len(x["p"]))
        out[m] = cb
    return out


def main() -> None:
    mask = load_core_mask()
    block_of, variant_of = load_block_variant()
    per_model = load_per_model(mask)
    cb = contrast_per_model_block(per_model, block_of, variant_of)

    models = sorted(per_model, key=model_sort_key)
    all_blocks = sorted({b for m in cb for b in cb[m]})

    per_model_mean = {m: statistics.mean(cb[m].values()) * 100 for m in models if cb[m]}
    median_over_models = statistics.median(per_model_mean.values())
    mean_over_models = statistics.mean(per_model_mean.values())

    # Cluster bootstrap: resample whole blocks, recompute per-model means and the
    # mean-over-models, collect the distribution.
    rng = random.Random(SEED)
    boot_pooled: List[float] = []
    boot_per_model: Dict[str, List[float]] = {m: [] for m in models}
    n_blocks = len(all_blocks)
    for _ in range(N_RESAMPLES):
        sample = [all_blocks[rng.randrange(n_blocks)] for _ in range(n_blocks)]
        pm: Dict[str, float] = {}
        for m in models:
            vals = [cb[m][b] for b in sample if b in cb[m]]
            if vals:
                pm[m] = sum(vals) / len(vals) * 100
        if pm:
            boot_pooled.append(statistics.mean(pm.values()))
        for m, v in pm.items():
            boot_per_model[m].append(v)

    def pct(xs: List[float], q: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]

    pooled_ci = (pct(boot_pooled, 0.025), pct(boot_pooled, 0.975))
    per_model_ci = {m: (pct(boot_per_model[m], 0.025), pct(boot_per_model[m], 0.975))
                    for m in models if boot_per_model[m]}

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "bootstrap": "block-level cluster bootstrap (resamples whole scenario-context blocks)",
        "n_blocks": n_blocks,
        "n_models": len(models),
        "contrast_definition": "per (model, block): direct_acc - mean(pragmatic_acc); summarised across models",
        "median_over_models_pp": round(median_over_models, 3),
        "mean_over_models_pp": round(mean_over_models, 3),
        "mean_over_models_ci95_pp": [round(pooled_ci[0], 3), round(pooled_ci[1], 3)],
        "per_model": {
            model_display(m): {
                "mean_pp": round(per_model_mean[m], 3),
                "ci95_pp": [round(per_model_ci[m][0], 3), round(per_model_ci[m][1], 3)],
                "n_blocks": len(cb[m]),
            }
            for m in models if m in per_model_mean
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"blocks={n_blocks}  models={len(models)}")
    print("per-model within-block mean contrast (pp), with cluster-bootstrap 95% CI:")
    for m in models:
        if m in per_model_mean:
            lo, hi = per_model_ci[m]
            print(f"  {model_display(m):26s} {per_model_mean[m]:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    print(f"\nmedian over models : {median_over_models:+.3f} pp")
    print(f"mean   over models : {mean_over_models:+.3f} pp  CI95 [{pooled_ci[0]:+.3f}, {pooled_ci[1]:+.3f}]")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
