"""
PCP Evaluation Pipeline
对多个LLM在PCP数据集上进行评估
支持本地模型 (transformers/vLLM) 和 API 模型 (OpenAI/Anthropic/DeepSeek)
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import parse_choice_answer


# ============================================================
# Model runners
# ============================================================

class BaseModelRunner:
    """Base class for model evaluation."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise NotImplementedError

    def run_intention_recognition(self, task: Dict) -> Dict:
        """Run multiple-choice intention recognition."""
        result = task.copy()
        response = self.generate(task["prompt"], max_tokens=10)

        result["model_answer"] = parse_choice_answer(response) or "?"

        result["model_raw_response"] = response.strip()
        result["parse_failed"] = result["model_answer"] == "?"
        result["correct"] = (result["model_answer"] == task["correct_letter"])
        return result

    def run_intention_generation(self, task: Dict) -> Dict:
        """Run free-text intention generation."""
        result = task.copy()
        response = self.generate(task["prompt"], max_tokens=256)
        result["model_response"] = response.strip()
        result["generation_reference_intent"] = task["generation_reference_intent"]
        return result


class TransformersRunner(BaseModelRunner):
    """Local model via Hugging Face Transformers."""

    def __init__(self, model_name: str, model_path: str, device: str = "cuda"):
        super().__init__(model_name)
        self.model_path = model_path
        self.device = device
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response


class APIRunner(BaseModelRunner):
    """API-based model runner (OpenAI, Anthropic, DeepSeek compatible)."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None,
                 provider: str = "openai"):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import urllib.request
        import urllib.error

        if self.provider == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
        elif self.base_url:
            url = self.base_url.rstrip("/") + "/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"

        data = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        try:
            resp = urllib.request.urlopen(req, timeout=60)
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            print(f"API error: {e.code} - {e.read().decode()}")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""


# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
    }

    # Per-domain accuracy
    domain_results = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        domain = r.get("domain", "unknown")
        domain_results[domain]["total"] += 1
        if r.get("correct"):
            domain_results[domain]["correct"] += 1

    for domain, counts in domain_results.items():
        metrics[f"accuracy_{domain}"] = (
            counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        )

    # Per-variant accuracy (direct vs. pragmatic variants)
    variant_results = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        variant = r.get("variant", "unknown")
        variant_results[variant]["total"] += 1
        if r.get("correct"):
            variant_results[variant]["correct"] += 1

    for variant, counts in variant_results.items():
        metrics[f"accuracy_{variant}"] = (
            counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        )

    # Per-phenomena accuracy
    phenomena_results = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        for ph in r.get("phenomena", []):
            phenomena_results[ph]["total"] += 1
            if r.get("correct"):
                phenomena_results[ph]["correct"] += 1

    for ph, counts in phenomena_results.items():
        metrics[f"accuracy_phenomenon_{ph}"] = (
            counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        )

    return metrics


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_model(
    runner: BaseModelRunner,
    tasks: List[Dict],
    task_type: str = "intention_recognition",
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict:
    """Run full evaluation for a model on the PCP dataset."""
    results = []
    samples = tasks[:max_samples] if max_samples else tasks

    print(f"\n{'='*60}")
    print(f"Evaluating {runner.model_name}")
    print(f"Task: {task_type}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}")

    for i, task in enumerate(samples):
        print(f"\r[{i+1}/{len(samples)}] Processing {task.get('probe_id', '?')}...", end="", flush=True)

        try:
            if task_type == "intention_recognition":
                result = runner.run_intention_recognition(task)
            elif task_type == "intention_generation":
                result = runner.run_intention_generation(task)
            else:
                result = task

            results.append(result)
        except Exception as e:
            print(f"\nError on {task.get('probe_id', '?')}: {e}")
            results.append({**task, "error": str(e)})

        # Rate limiting for APIs
        if isinstance(runner, APIRunner):
            time.sleep(0.5)

    print()  # newline after progress

    # Compute metrics
    if task_type == "intention_recognition":
        metrics = compute_metrics(results)
    else:
        metrics = {"total": len(results)}

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_slug = runner.model_name.replace("/", "_").replace(":", "_")

        with open(out_path / f"results_{model_slug}_{task_type}.json", "w", encoding="utf-8") as f:
            json.dump({"model": runner.model_name, "task_type": task_type, "metrics": metrics, "results": results},
                      f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n--- {runner.model_name} Results ---")
    if "accuracy" in metrics:
        print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        for k, v in sorted(metrics.items()):
            if k.startswith("accuracy_") and k != "accuracy":
                print(f"  {k}: {v:.2%}")

    return {"model": runner.model_name, "metrics": metrics, "results": results}


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PCP Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path (for transformers)")
    parser.add_argument("--provider", type=str, choices=["api", "local"], default="api")
    parser.add_argument("--api_provider", type=str, choices=["deepseek", "openai", "anthropic"], default="deepseek")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL")
    parser.add_argument("--task_type", type=str, default="intention_recognition",
                        choices=["intention_recognition", "intention_generation"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load tasks
    tasks_file = Path(args.data_dir) / "pcp_tasks.json"
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data["tasks"][args.task_type]
    print(f"Loaded {len(tasks)} {args.task_type} tasks")

    # Create runner
    if args.provider == "local":
        runner = TransformersRunner(args.model, args.model_path, args.device)
    else:
        api_key = args.api_key
        if not api_key:
            raise SystemExit("API key is required; pass --api_key or use run_api_upper_bounds.py.")
        runner = APIRunner(args.model, api_key, args.base_url, args.api_provider)

    # Run evaluation
    evaluate_model(runner, tasks, args.task_type, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
