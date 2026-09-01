"""
Open-weight local model evaluator for PCP recognition tasks.

This runner is intentionally independent from the API experiment scripts. It
supports repeated shuffled option orders, multiple prompt styles, majority
aggregation, raw responses, parse failure accounting, latency, and environment
records for single-GPU server runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import (  # noqa: E402
    compute_metrics,
    load_compact_tasks,
    majority_vote_by_probe,
    make_shuffled_choice_tasks,
    parse_choice_answer,
    write_json,
)


DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
]

DEFAULT_SPLITS = {
    "PCP-Hard-500-final": "data/processed/pcp_hard_500_final.json",
    "PCP-Core-3000-final": "data/processed/pcp_core_3000_final.json",
}


def slugify(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_ENDPOINT",
                "TRANSFORMERS_CACHE",
                "HF_HUB_OFFLINE",
            ]
            if os.environ.get(key) is not None
        },
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_error"] = repr(exc)

    for package in ["transformers", "accelerate", "bitsandbytes", "vllm"]:
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            info[f"{package}_error"] = repr(exc)

    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        info["nvidia_smi"] = proc.stdout
    except Exception as exc:
        info["nvidia_smi_error"] = repr(exc)
    return info


class TransformersChatRunner:
    def __init__(self, model_id: str, load_mode: str, max_new_tokens: int):
        self.model_id = model_id
        self.load_mode = load_mode
        self.max_new_tokens = max_new_tokens
        self.actual_load_mode = None
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        modes = [self.load_mode]
        if self.load_mode == "auto":
            modes = ["bf16", "fp16", "4bit"]

        errors = []
        for mode in modes:
            try:
                kwargs: Dict[str, Any] = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if mode == "bf16":
                    kwargs["torch_dtype"] = torch.bfloat16
                elif mode == "fp16":
                    kwargs["torch_dtype"] = torch.float16
                elif mode == "4bit":
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    raise ValueError(f"Unknown load mode: {mode}")

                print(f"Loading {self.model_id} with mode={mode}", flush=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                self.model.eval()
                self.actual_load_mode = mode
                return
            except Exception as exc:
                errors.append({"mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
                print(f"Load failed for {self.model_id} mode={mode}: {exc}", flush=True)
                try:
                    del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        raise RuntimeError(json.dumps(errors, ensure_ascii=False, indent=2))

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        import torch

        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - t0
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        return response, latency


def maybe_subset(tasks: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if not max_samples or max_samples >= len(tasks):
        return tasks
    rng = random.Random(seed)
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        by_variant.setdefault(task.get("variant", "unknown"), []).append(task)

    selected: List[Dict[str, Any]] = []
    remaining = max_samples
    variants = sorted(by_variant)
    for idx, variant in enumerate(variants):
        bucket = list(by_variant[variant])
        rng.shuffle(bucket)
        take = remaining if idx == len(variants) - 1 else max(1, round(max_samples * len(bucket) / len(tasks)))
        take = min(take, len(bucket), remaining)
        selected.extend(bucket[:take])
        remaining -= take
        if remaining <= 0:
            break
    rng.shuffle(selected)
    return selected[:max_samples]


def run_split(
    runner: TransformersChatRunner,
    model_id: str,
    split_name: str,
    tasks_file: Path,
    output_dir: Path,
    seeds: Iterable[int],
    prompt_styles: Iterable[str],
    max_samples: Optional[int],
    subset_seed: int,
) -> Dict[str, Any]:
    compact = load_compact_tasks(tasks_file)
    base_tasks = compact["recognition"]
    base_tasks = maybe_subset(base_tasks, max_samples, subset_seed)
    eval_tasks = make_shuffled_choice_tasks(base_tasks, list(seeds), list(prompt_styles))

    results: List[Dict[str, Any]] = []
    correct = 0
    parse_failures = 0
    t_start = time.time()
    model_slug = slugify(model_id)
    split_slug = slugify(split_name)

    raw_path = output_dir / f"{model_slug}__{split_slug}__recognition_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for i, task in enumerate(eval_tasks, start=1):
            item = {
                "eval_id": task.get("eval_id"),
                "probe_id": task["probe_id"],
                "domain": task.get("domain"),
                "variant": task.get("variant"),
                "template_id": task.get("template_id"),
                "phenomena": task.get("phenomena", []),
                "shuffle_seed": task.get("shuffle_seed"),
                "prompt_style": task.get("prompt_style"),
                "correct_letter": task["correct_letter"],
                "correct_idx": task.get("correct_idx"),
                "recognition_gold_choice": task.get("recognition_gold_choice"),
            }
            try:
                response, latency = runner.generate(task["prompt"])
                answer = parse_choice_answer(response) or "?"
                answer_idx = ord(answer) - ord("A") if answer in {"A", "B", "C", "D"} else None
                model_choice = task["choices"][answer_idx] if answer_idx is not None else None
                item.update(
                    {
                        "model_answer": answer,
                        "model_choice": model_choice,
                        "correct": answer == task["correct_letter"],
                        "parse_failed": answer == "?",
                        "raw_response": response,
                        "latency_seconds": latency,
                    }
                )
            except Exception as exc:
                item.update(
                    {
                        "model_answer": "?",
                        "correct": False,
                        "parse_failed": True,
                        "raw_response": "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

            correct += int(item["correct"])
            parse_failures += int(item["parse_failed"])
            results.append(item)
            raw_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw_file.flush()

            if i % 50 == 0 or i == len(eval_tasks):
                elapsed = time.time() - t_start
                print(
                    f"{model_id} {split_name} [{i}/{len(eval_tasks)}] "
                    f"acc={correct/i:.2%} parse_fail={parse_failures/i:.2%} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    metrics = compute_metrics(results)
    majority_results = majority_vote_by_probe(results)
    majority_metrics = compute_metrics(majority_results)
    semantic_majority_results = semantic_majority_vote(results)
    semantic_majority_metrics = compute_metrics(semantic_majority_results)
    elapsed = time.time() - t_start

    parse_failure_items = [r for r in results if r.get("parse_failed") or r.get("error")]
    output = {
        "model": model_id,
        "actual_load_mode": runner.actual_load_mode,
        "split": split_name,
        "tasks_file": str(tasks_file),
        "recognition_base_total": len(base_tasks),
        "recognition_eval_total": len(eval_tasks),
        "seeds": list(seeds),
        "prompt_styles": list(prompt_styles),
        "is_full_split": max_samples is None or max_samples >= len(compact["recognition"]),
        "max_samples": max_samples,
        "time_seconds": elapsed,
        "metrics": metrics,
        "majority_metrics": majority_metrics,
        "semantic_majority_metrics": semantic_majority_metrics,
        "parse_failures": parse_failure_items,
        "results": results,
        "majority_results": majority_results,
        "semantic_majority_results": semantic_majority_results,
        "raw_jsonl": str(raw_path),
    }
    out_path = output_dir / f"{model_slug}__{split_slug}__recognition.json"
    write_json(out_path, output)
    print(f"Saved {out_path}", flush=True)
    return output


def semantic_majority_vote(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate repeated shuffled runs by selected option text, not letter."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["probe_id"]].append(result)

    aggregated = []
    for probe_id, probe_results in grouped.items():
        first = probe_results[0]
        valid_choices = [
            r.get("model_choice")
            for r in probe_results
            if r.get("model_choice") and not r.get("parse_failed")
        ]
        counts = Counter(valid_choices)
        if counts:
            model_choice, votes = counts.most_common(1)[0]
        else:
            model_choice, votes = None, 0
        aggregated.append(
            {
                "probe_id": probe_id,
                "domain": first.get("domain"),
                "variant": first.get("variant"),
                "phenomena": first.get("phenomena", []),
                "model_choice": model_choice,
                "recognition_gold_choice": first.get("recognition_gold_choice"),
                "correct": model_choice is not None
                and model_choice == first.get("recognition_gold_choice"),
                "votes": votes,
                "runs": len(probe_results),
                "parse_failed": model_choice is None,
            }
        )
    return aggregated


def write_failure(output_dir: Path, model_id: str, error: BaseException) -> None:
    payload = {
        "model": model_id,
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "environment": environment_info(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_dir / f"{slugify(model_id)}__FAILED.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--splits", default="PCP-Hard-500-final,PCP-Core-3000-final")
    parser.add_argument("--output_dir", default="experiments/open_models_worker1")
    parser.add_argument("--load_mode", choices=["auto", "bf16", "fp16", "4bit"], default="auto")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--prompt_styles", default="letter,json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--core_max_samples", type=int, default=None)
    parser.add_argument("--hard_max_samples", type=int, default=None)
    parser.add_argument("--subset_seed", type=int, default=20260506)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--hf_endpoint", default=None)
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", environment_info())

    models = parse_csv(args.models)
    splits = parse_csv(args.splits)
    seeds = parse_int_csv(args.seeds)
    prompt_styles = parse_csv(args.prompt_styles)

    summary: Dict[str, Any] = {"runs": [], "failures": []}
    for model_id in models:
        try:
            runner = TransformersChatRunner(model_id, args.load_mode, args.max_new_tokens)
        except Exception as exc:
            print(f"FAILED to load {model_id}: {exc}", flush=True)
            write_failure(output_dir, model_id, exc)
            summary["failures"].append({"model": model_id, "stage": "load", "error": repr(exc)})
            continue

        for split_name in splits:
            tasks_file = Path(DEFAULT_SPLITS[split_name])
            split_slug = slugify(split_name)
            model_slug = slugify(model_id)
            out_path = output_dir / f"{model_slug}__{split_slug}__recognition.json"
            if args.skip_existing and out_path.exists():
                print(f"Skipping existing {out_path}", flush=True)
                continue
            max_samples = args.max_samples
            if split_name in {
                "PCP-Hard-500-final",
            } and args.hard_max_samples is not None:
                max_samples = args.hard_max_samples
            if split_name in {"PCP-Core-3000-final"} and args.core_max_samples is not None:
                max_samples = args.core_max_samples

            try:
                result = run_split(
                    runner=runner,
                    model_id=model_id,
                    split_name=split_name,
                    tasks_file=tasks_file,
                    output_dir=output_dir,
                    seeds=seeds,
                    prompt_styles=prompt_styles,
                    max_samples=max_samples,
                    subset_seed=args.subset_seed,
                )
                summary["runs"].append(
                    {
                        "model": model_id,
                        "split": split_name,
                        "total": result["metrics"]["total"],
                        "accuracy": result["metrics"]["accuracy"],
                        "parse_failure_rate": result["metrics"]["parse_failure_rate"],
                        "majority_accuracy": result["majority_metrics"]["accuracy"],
                        "majority_parse_failure_rate": result["majority_metrics"]["parse_failure_rate"],
                        "is_full_split": result["is_full_split"],
                        "output": str(out_path),
                    }
                )
                write_json(output_dir / "summary.json", summary)
            except Exception as exc:
                print(f"FAILED run {model_id} {split_name}: {exc}", flush=True)
                write_failure(output_dir, f"{model_id}__{split_name}", exc)
                summary["failures"].append(
                    {"model": model_id, "split": split_name, "stage": "run", "error": repr(exc)}
                )
                write_json(output_dir / "summary.json", summary)

        try:
            del runner
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
