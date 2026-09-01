"""
Local API-based evaluation runner.
Uses DeepSeek API (or other OpenAI-compatible APIs) to evaluate PCP tasks.
No GPU required — runs from local machine.
"""

import json
import time
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcp_eval_utils import parse_choice_answer


class APIClient:
    """Generic OpenAI-compatible API client."""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.model = model
    
    def chat(self, prompt: str, max_tokens: int = 10, temperature: float = 0.1) -> str:
        url = f"{self.base_url}/chat/completions"
        
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            print(f"  API Error {e.code}: {err[:200]}")
            return ""
        except Exception as e:
            print(f"  Error: {e}")
            return ""


def run_intention_recognition(client: APIClient, tasks: list) -> tuple:
    """Run multiple-choice intention recognition."""
    results = []
    correct = 0
    
    for i, task in enumerate(tasks):
        response = client.chat(task["prompt"], max_tokens=10)
        
        answer = parse_choice_answer(response) or "?"
        
        is_correct = (answer == task["correct_letter"])
        if is_correct:
            correct += 1
        
        results.append({
            "probe_id": task["probe_id"],
            "domain": task["domain"],
            "variant": task["variant"],
            "phenomena": task["phenomena"],
            "model_answer": answer,
            "correct_letter": task["correct_letter"],
            "correct": is_correct,
            "parse_failed": answer == "?",
            "raw_response": response.strip()[:100],
        })
        
        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(tasks)}] Accuracy: {acc:.2%}")
        
        # Rate limiting
        time.sleep(0.3)
    
    accuracy = correct / len(tasks) if tasks else 0
    return results, accuracy


def compute_metrics(results: list, model_name: str) -> dict:
    """Compute detailed metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    
    metrics = {
        "model_name": model_name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
    }
    
    # Per domain
    for domain in ["emergency", "primary_care", "mental_health", "specialist", "pediatric", "geriatric"]:
        subset = [r for r in results if r["domain"] == domain]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            metrics[f"accuracy_{domain}"] = c / len(subset)
    
    # Per variant — group into meta-categories
    variant_groups = {
        "direct": ["direct"],
        "pragmatic": [],  # all non-direct variants
    }
    
    direct_results = [r for r in results if r["variant"] == "direct"]
    pragmatic_results = [r for r in results if r["variant"] != "direct"]
    
    if direct_results:
        metrics["accuracy_variant_direct"] = sum(1 for r in direct_results if r["correct"]) / len(direct_results)
    if pragmatic_results:
        metrics["accuracy_variant_pragmatic"] = sum(1 for r in pragmatic_results if r["correct"]) / len(pragmatic_results)
        # Pragmatic gap
        if direct_results:
            metrics["pragmatic_gap"] = metrics["accuracy_variant_direct"] - metrics["accuracy_variant_pragmatic"]
    
    # Per phenomenon
    phenom_results = defaultdict(list)
    for r in results:
        for ph in r["phenomena"]:
            phenom_results[ph].append(r)
    
    for ph, subset in phenom_results.items():
        c = sum(1 for r in subset if r["correct"])
        metrics[f"accuracy_phen_{ph}"] = c / len(subset)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--tasks_file", type=str, default="data/processed/pcp_compact.json")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("API key is required; pass --api_key or use run_api_upper_bounds.py.")
    
    # Load tasks
    with open(args.tasks_file, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)
    
    tasks = tasks_data["recognition"]
    if args.max_samples:
        tasks = tasks[:args.max_samples]
    
    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks)}")
    print(f"{'='*60}")
    
    # Create client
    client = APIClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    
    # Run evaluation
    t0 = time.time()
    results, accuracy = run_intention_recognition(client, tasks)
    elapsed = time.time() - t0
    
    # Compute metrics
    metrics = compute_metrics(results, args.model)
    metrics["time_seconds"] = elapsed
    metrics["time_per_sample"] = elapsed / len(tasks) if tasks else 0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s/sample)")
    
    # Pragmatic gap
    if "pragmatic_gap" in metrics:
        print(f"\n*** PRAGMATIC GAP: {metrics['pragmatic_gap']:.2%} ***")
        print(f"  Direct utterances:     {metrics.get('accuracy_variant_direct', 0):.2%}")
        print(f"  Pragmatic utterances:  {metrics.get('accuracy_variant_pragmatic', 0):.2%}")
    
    print(f"\nPer domain:")
    for k, v in sorted(metrics.items()):
        if k.startswith("accuracy_") and "variant" not in k and "phen" not in k and k != "accuracy":
            print(f"  {k.replace('accuracy_', ''):20s}: {v:.2%}")
    
    print(f"\nPer pragmatic phenomenon:")
    for k, v in sorted(metrics.items()):
        if k.startswith("accuracy_phen_"):
            print(f"  {k.replace('accuracy_phen_', ''):20s}: {v:.2%}")
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    output_file = Path(args.output_dir) / f"{model_slug}_results.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {output_file}")
    return metrics


if __name__ == "__main__":
    main()
