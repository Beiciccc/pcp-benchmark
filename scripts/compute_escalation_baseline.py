#!/usr/bin/env python3
"""Clinical escalation-keyword baseline (Appendix D, "Answer-Choice Integrity
and Shortcut Baselines").

The original baseline was an ad-hoc script that was not retained; this file
reconstructs it from the recipe documented verbatim in the paper so that every
escalation figure is reproducible and pinned. Each answer option is scored by

    score(option) = (# escalation cues) - (# de-escalation cues)

over the documented lexicon, counting a cue when it is the prefix of a word in
the option (so "evaluate" matches "evaluation", "screen" matches "screening").
The baseline selects a most-escalation option (ties broken at random). This
word-prefix rule reproduces the paper's 38.4% unique-most-escalation figure
exactly (161/419 on Hard pragmatic items); tied-at-max is 42.0% (176/419) under
the same rule, and the headline accuracy is tie-break-dependent (random-tie
expected value 52.8%, first-among-ties 54.2%), bracketing the 52.6% in the paper.

We report, per split, over the pragmatic items (variant != direct):
* unique-most-escalation rate (gold is the sole argmax),
* tied-at-max rate (gold is among >1 argmax),
* accuracy under deterministic first-among-ties selection,
* accuracy under random tie-breaking (expected value, plus worst/best range).

Run: python3 scripts/compute_escalation_baseline.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_PATH = ROOT / "experiments" / "model_results" / "escalation_baseline.json"

SPLITS = {
    "hard": PROCESSED / "pcp_hard_469_signoff_final.json",
    "core": PROCESSED / "pcp_core_3000_final.json",
    "real_clean": PROCESSED / "pcp_real_clean_64_signoff_final.json",
}

# Lexicon documented in the paper (Appendix D).
ESC = ["urgent", "evaluate", "assess", "screen", "risk", "safety"]
DEESC = ["monitor", "routine", "reassurance", "follow-up"]


def score(option: str) -> int:
    words = re.findall(r"[a-z]+", option.lower())
    esc = sum(1 for w in words for c in ESC if w.startswith(c))
    deesc = sum(1 for w in words for c in DEESC if w.startswith(c.replace("-", "")))
    return esc - deesc


def analyse(items):
    n = uniq = tied = acc_first = 0
    acc_ev = 0.0
    for t in items:
        choices = t.get("choices")
        gi = t.get("correct_idx")
        if not choices or gi is None:
            continue
        n += 1
        sc = [score(c) for c in choices]
        mx = max(sc)
        max_idx = [i for i, s in enumerate(sc) if s == mx]
        if max_idx[0] == gi:
            acc_first += 1
        if gi in max_idx:
            if len(max_idx) == 1:
                uniq += 1
            else:
                tied += 1
            acc_ev += 1.0 / len(max_idx)
    gold_at_max = uniq + tied
    return {
        "n": n,
        "unique_most_escalation": uniq,
        "unique_pct": round(100 * uniq / n, 1) if n else 0.0,
        "tied_at_max": tied,
        "tied_pct": round(100 * tied / n, 1) if n else 0.0,
        "gold_at_max_pct": round(100 * gold_at_max / n, 1) if n else 0.0,
        "accuracy_first_tie_pct": round(100 * acc_first / n, 1) if n else 0.0,
        "accuracy_random_tiebreak_ev_pct": round(100 * acc_ev / n, 1) if n else 0.0,
        "accuracy_random_tiebreak_worst_pct": round(100 * uniq / n, 1) if n else 0.0,
        "accuracy_random_tiebreak_best_pct": round(100 * gold_at_max / n, 1) if n else 0.0,
    }


def stratify_hard_accuracy():
    """Per-model Hard-pragmatic accuracy stratified by the escalation shortcut.

    Splits the 419 Hard pragmatic items into three strata: the gold option is the
    unique argmax of the escalation score, is tied at the argmax, or is not at the
    argmax at all. The last stratum is the decisive one: there the conservatism
    shortcut actively selects a wrong option, so a model relying on it would score
    near zero. Returns {display_model: {stratum: accuracy}} plus pooled values.
    """
    import csv as _csv
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from pcp_eval_utils import majority_vote_by_probe
    from summarize_final_model_results import dedupe_api_results, model_display, model_sort_key

    d = json.loads(SPLITS["hard"].read_text(encoding="utf-8"))
    strata = {}
    for t in d["tasks"]["recognition"]:
        if t.get("variant") == "direct":
            continue
        sc = [score(c) for c in t["choices"]]
        mx = max(sc)
        max_idx = [i for i, s in enumerate(sc) if s == mx]
        gi = t["correct_idx"]
        strata[t["probe_id"]] = ("unique_max" if (gi in max_idx and len(max_idx) == 1)
                                 else ("tied_max" if gi in max_idx else "not_max"))

    mask = set()
    mp = ROOT / "data" / "validation" / "final_inclusion_mask.csv"
    if mp.exists():
        for r in _csv.DictReader(mp.open(encoding="utf-8-sig")):
            if (r.get("included_in_main_eval", "").strip().lower() == "true"
                    and r.get("split", "").strip().lower() == "hard"):
                mask.add(r["probe_id"])

    keys = ["unique_max", "tied_max", "not_max"]
    per_model, pooled = {}, {k: [0, 0] for k in keys}
    for path in sorted((ROOT / "experiments" / "api_hard_469_final").glob("*_pcp_full.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model") or path.stem
        rows = payload.get("results", {}).get("recognition") or []
        if not rows:
            continue
        corr = {r["probe_id"]: bool(r.get("correct"))
                for r in majority_vote_by_probe(dedupe_api_results(rows, "recognition"))
                if "probe_id" in r}
        cell = {}
        for k in keys:
            ids = [i for i, s in strata.items() if s == k and i in corr and (not mask or i in mask)]
            c = sum(corr[i] for i in ids)
            cell[k] = round(100 * c / len(ids), 1) if ids else None
            pooled[k][0] += c
            pooled[k][1] += len(ids)
        per_model[model_display(model)] = cell
    return {
        "stratum_sizes": {k: sum(1 for v in strata.values() if v == k) for k in keys},
        "per_model_accuracy_pct": per_model,
        "pooled_accuracy_pct": {k: round(100 * pooled[k][0] / pooled[k][1], 1) for k in keys if pooled[k][1]},
    }


def main() -> None:
    out = {"created_at": datetime.now(timezone.utc).isoformat(),
           "lexicon_escalation": ESC, "lexicon_deescalation": DEESC,
           "matching": "word-prefix", "splits": {}}
    for split, path in SPLITS.items():
        d = json.loads(path.read_text(encoding="utf-8"))
        rec = d["tasks"]["recognition"]
        prag = [t for t in rec if t.get("variant") != "direct"]
        out["splits"][split] = analyse(prag)
    out["hard_stratified_accuracy"] = stratify_hard_accuracy()
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for split, r in out["splits"].items():
        print(f"=== {split} (pragmatic n={r['n']}) ===")
        print(f"  unique-most-escalation : {r['unique_most_escalation']} ({r['unique_pct']}%)")
        print(f"  tied-at-max            : {r['tied_at_max']} ({r['tied_pct']}%)")
        print(f"  gold-at-max total      : {r['gold_at_max_pct']}%")
        print(f"  accuracy (first-tie)   : {r['accuracy_first_tie_pct']}%")
        print(f"  accuracy (random EV)   : {r['accuracy_random_tiebreak_ev_pct']}%"
              f"  [range {r['accuracy_random_tiebreak_worst_pct']}-{r['accuracy_random_tiebreak_best_pct']}%]")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
