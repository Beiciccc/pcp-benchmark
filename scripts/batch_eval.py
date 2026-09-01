#!/usr/bin/env python3
"""
Batch evaluation orchestrator — runs all models sequentially, collects results.
To be run on the GPU server.
"""

import json
import time
import subprocess
import sys
from pathlib import Path

MODELS = [
    {
        "name": "Qwen2.5-7B",
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5 7B Instruct",
    },
    {
        "name": "Mistral-7B",
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct v0.3",
    },
    {
        "name": "Llama-3-8B",
        "path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Meta Llama 3 8B Instruct",
    },
]

RESULTS_DIR = "results"
TASKS_FILE = "pcp_compact.json"

def run_model(model_config):
    """Run evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Starting: {model_config['description']}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "run_server_eval.py",
        "--model_path", model_config["path"],
        "--model_name", model_config["name"],
        "--tasks_file", TASKS_FILE,
        "--output_dir", RESULTS_DIR,
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    
    if result.returncode != 0:
        print(f"ERROR: {model_config['name']} failed with code {result.returncode}")
        return None
    
    # Load and return results
    result_file = Path(RESULTS_DIR) / f"{model_config['name']}_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
        acc = data['metrics']['accuracy']
        print(f"\n>>> {model_config['name']}: {acc:.2%} (took {elapsed:.0f}s)")
        return data
    
    return None

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    all_results = {}
    
    for model in MODELS:
        try:
            result = run_model(model)
            if result:
                all_results[model['name']] = result
        except Exception as e:
            print(f"Failed to run {model['name']}: {e}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20s} {'Accuracy':>10s} {'Time':>10s}")
    print("-"*40)
    
    for name, data in all_results.items():
        acc = data['metrics']['accuracy']
        t = data['metrics'].get('time_seconds', 0)
        print(f"{name:<20s} {acc:>9.2%} {t:>9.0f}s")
    
    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": all_results,
    }
    with open(Path(RESULTS_DIR) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to {RESULTS_DIR}/summary.json")

if __name__ == "__main__":
    main()
