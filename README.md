# PCP: A Controlled Benchmark for Pragmatic Intention Recognition in Clinical Communication

PCP is a 3,533-item multiple-choice benchmark that tests whether a model can recover the
clinical action a speaker wants a clinician to take when that intent is expressed
indirectly: through hedging, indirect speech acts, urgency cues, politeness, or implicature.

Accepted for oral presentation at **LUHME 2026** (3rd Workshop on Language Understanding in
the Human-Machine Era), co-located with EMNLP 2026, Budapest.

## Splits

| Split | Items | Description |
|---|---|---|
| `pcp_core_3000.json` | 3,000 | Controlled synthetic probes: 300 scenario-context blocks, each with one `direct` item and nine pragmatic variants of the same content. |
| `pcp_hard_469.json` | 469 | Close-distractor split (419 pragmatic + 50 direct controls). Each pragmatic item carries three tagged close distractors: `same_target_wrong_action`, `adjacent_target_same_action`, `wrong_pragmatic_focus`. |
| `pcp_real_clean_64.json` | 64 | Externally grounded split, derived by de-identified paraphrase from the public MTS-Dialog and ACI-Bench doctor-patient dialogue corpora. |

`final_inclusion_mask.csv` marks the rows included in the reported evaluation.

## Main finding

Across eleven systems (six API-accessed, five open-weight), ten select the intended clinical
action at 95-98% aggregate accuracy but name the pragmatic phenomenon licensing it at only
56-65%: a 30-40 pp within-item dissociation. Models act on a pragmatic cue without reliably
naming the mechanism behind it.

Accuracy is near ceiling on Core (98.7-99.8% for API models) but drops on Hard (83.4-90.6%)
and Real-Clean (79.7-90.6%).

## Repository layout

- `data/` - the three benchmark splits and the final inclusion mask.
- `results/` - derived result summaries: per-model accuracy, bootstrap CIs, pairwise McNemar
  tests, the within-block Core contrast, the escalation-keyword shortcut analysis, phenomenon
  classification, and generation-judge summaries.
- `scripts/` - evaluation and aggregation code, including the three analyses used for the
  camera-ready: `compute_bootstrap_ci_and_mcnemar.py`, `compute_within_block_ci.py`, and
  `compute_escalation_baseline.py`.

## Deliberately not included

- **Raw source dialogues** from MTS-Dialog and ACI-Bench. These are not redistributed; obtain
  them from their original repositories. Per-item provenance is recorded in the Real-Clean
  rows, with source records identified by HMAC so provenance is verifiable without exposing
  the underlying record.
- **Raw per-call model outputs.** The released `results/` files are the derived summaries the
  paper reports; full model-row recomputation requires rerunning the documented endpoints.
- **Annotation administration**: participant- and reviewer-facing packets, identifier
  mappings, recruitment records, and compensation records.

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Real-Clean split
is derived from MTS-Dialog and ACI-Bench, both distributed under CC BY 4.0; attribution to
those corpora is recorded per row and must be preserved in downstream use.

## Citation

See `CITATION.cff`.
