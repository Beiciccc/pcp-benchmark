"""
Server-side PCP evaluation runner for Windows GPU server.
Evaluates local models (Llama, Mistral, Qwen) on PCP tasks.
"""

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict


def load_tasks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_intention_recognition(model, tokenizer, tasks, device='cuda'):
    """Run multiple-choice intention recognition."""
    import torch
    
    results = []
    correct = 0
    
    for i, task in enumerate(tasks):
        prompt = task['prompt']
        
        messages = [{'role': 'user', 'content': prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = prompt
        
        inputs = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        response_upper = response.strip().upper()
        answer = '?'
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                answer = letter
                break
        
        is_correct = (answer == task['correct_letter'])
        if is_correct:
            correct += 1
        
        results.append({
            'probe_id': task['probe_id'],
            'domain': task['domain'],
            'variant': task['variant'],
            'phenomena': task['phenomena'],
            'model_answer': answer,
            'correct_letter': task['correct_letter'],
            'correct': is_correct,
            'raw_response': response.strip()[:100],
        })
        
        if (i + 1) % 25 == 0:
            print(f'  [{i+1}/{len(tasks)}] Accuracy so far: {correct/(i+1):.2%}')
    
    accuracy = correct / len(tasks)
    return results, accuracy


def compute_metrics(results):
    """Compute detailed metrics."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    metrics = {
        'total': total,
        'correct': correct,
        'accuracy': correct / total,
    }
    
    # Per domain
    domain_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        domain_stats[r['domain']]['total'] += 1
        if r['correct']:
            domain_stats[r['domain']]['correct'] += 1
    for d, s in domain_stats.items():
        metrics[f'accuracy_{d}'] = s['correct'] / s['total']
    
    # Per variant
    variant_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        variant_stats[r['variant']]['total'] += 1
        if r['correct']:
            variant_stats[r['variant']]['correct'] += 1
    for v, s in variant_stats.items():
        metrics[f'accuracy_variant_{v}'] = s['correct'] / s['total']
    
    # Per phenomenon
    phenom_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        for ph in r['phenomena']:
            phenom_stats[ph]['total'] += 1
            if r['correct']:
                phenom_stats[ph]['correct'] += 1
    for ph, s in phenom_stats.items():
        metrics[f'accuracy_phen_{ph}'] = s['correct'] / s['total']
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='HuggingFace model path or name')
    parser.add_argument('--model_name', type=str, required=True, help='Display name')
    parser.add_argument('--tasks_file', type=str, default='pcp_compact.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--load_in_4bit', action='store_true', default=True)
    args = parser.parse_args()
    
    print(f'Loading model: {args.model_path}')
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Load tasks
    tasks_data = load_tasks(args.tasks_file)
    tasks = tasks_data['recognition'][:args.max_samples] if args.max_samples else tasks_data['recognition']
    
    print(f'Running {len(tasks)} tasks...')
    t0 = time.time()
    results, accuracy = run_intention_recognition(model, tokenizer, tasks)
    elapsed = time.time() - t0
    
    # Compute metrics
    metrics = compute_metrics(results)
    metrics['model_name'] = args.model_name
    metrics['model_path'] = args.model_path
    metrics['time_seconds'] = elapsed
    metrics['time_per_sample'] = elapsed / len(tasks)
    
    print(f'\n=== {args.model_name} Results ===')
    print(f'Accuracy: {metrics["accuracy"]:.2%} ({metrics["correct"]}/{metrics["total"]})')
    print(f'Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s/sample)')
    print(f'\nPer domain:')
    for k, v in sorted(metrics.items()):
        if k.startswith('accuracy_') and not k.startswith('accuracy_variant') and not k.startswith('accuracy_phen'):
            print(f'  {k}: {v:.2%}')
    
    print(f'\nPer variant:')
    for k, v in sorted(metrics.items()):
        if k.startswith('accuracy_variant'):
            print(f'  {k}: {v:.2%}')
    
    print(f'\nPer phenomenon:')
    for k, v in sorted(metrics.items()):
        if k.startswith('accuracy_phen'):
            print(f'  {k}: {v:.2%}')
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_dir) / f'{args.model_name}_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f'\nResults saved to: {output_file}')
    return metrics


if __name__ == '__main__':
    main()
