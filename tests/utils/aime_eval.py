"""
AIME Dataset Evaluation Module

This module provides functions to evaluate language models on the combined AIME dataset
(test2024 + test2025-I + test2025-II).
"""

import json
import requests
import os
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import SamplingParams


def download_and_combine_aime_datasets(data_dir: str = "./data/aime") -> str:
    """Download all AIME datasets and combine them into a single file"""

    datasets = {
        "test2024": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2024.jsonl",
        "test2025-I": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-I.jsonl",
        "test2025-II": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-II.jsonl",
    }

    os.makedirs(data_dir, exist_ok = True)
    combined_filepath = os.path.join(data_dir, "aime.jsonl")

    # Check if combined file already exists
    if os.path.exists(combined_filepath):
        print(f"Combined AIME dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading and combining AIME datasets...")

    all_problems = []
    global_id = 0

    for dataset_name, url in datasets.items():
        print(f"  Downloading {dataset_name}...")

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse each line and add source information
            for line_num, line in enumerate(response.text.strip().split("\n")):
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Add source dataset information and global ID
                        data["source_dataset"] = dataset_name
                        data["original_id"] = data.get("id", line_num)
                        data["global_id"] = global_id
                        global_id += 1
                        all_problems.append(data)
                    except json.JSONDecodeError as e:
                        print(
                            f"    Warning: Error parsing line {line_num + 1} in {dataset_name}: {e}"
                        )
                        continue

        except requests.RequestException as e:
            print(f"    Error downloading {dataset_name}: {e}")
            continue

    # Write combined dataset
    if all_problems:
        with open(combined_filepath, "w", encoding = "utf-8") as f:
            for problem in all_problems:
                f.write(json.dumps(problem, ensure_ascii = False) + "\n")

        print(f"✅ Combined {len(all_problems)} problems from {len(datasets)} datasets")
        print(f"   Saved to: {combined_filepath}")

        # Print summary by dataset
        for dataset_name in datasets.keys():
            count = sum(1 for p in all_problems if p["source_dataset"] == dataset_name)
            print(f"   {dataset_name}: {count} problems")

    else:
        raise RuntimeError("No problems were successfully downloaded")

    return combined_filepath


def load_aime_dataset(data_dir: str = "./data/aime") -> List[Dict[str, Any]]:
    """Load combined AIME dataset and format for evaluation"""

    # Download and combine if needed
    filepath = download_and_combine_aime_datasets(data_dir)

    examples = []
    with open(filepath, "r", encoding = "utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)

                    # Format as expected by our evaluation
                    formatted_example = {
                        "global_id": data.get("global_id", line_num),
                        "original_id": data.get(
                            "original_id", data.get("id", line_num)
                        ),
                        "source_dataset": data.get("source_dataset", "unknown"),
                        "problem": data["problem"],
                        "answer": str(data["answer"]),  # Ensure answer is string
                        "solution": data.get("solution", ""),
                        "url": data.get("url", ""),
                        # Format as chat messages for the model
                        "prompt": [
                            {
                                "role": "system",
                                "content": "You are a mathematical problem solver. Solve the given problem step by step and provide your final answer clearly.",
                            },
                            {
                                "role": "user",
                                "content": f"Problem: {data['problem']}\n\nSolve this step by step and provide your final numerical answer.",
                            },
                        ],
                    }
                    examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(examples)} problems from combined AIME dataset")

    # Print breakdown by source
    source_counts = {}
    for example in examples:
        source = example["source_dataset"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")

    return examples


def extract_aime_answer(response: str) -> str:
    """Extract numerical answer from AIME response"""

    # AIME answers are integers from 0-999
    # Look for patterns like "The answer is 123" or just standalone numbers
    patterns = [
        r"(?:the )?(?:final )?answer is (\d{1,3})",
        r"(?:therefore|thus|so),?\s*(?:the )?(?:final )?answer is (\d{1,3})",
        r"\\boxed\{(\d{1,3})\}",
        r"\$\\boxed\{(\d{1,3})\}\$",
        r"(?:answer|result):\s*(\d{1,3})",
        r"(?:^|\n)\s*(\d{1,3})\s*(?:\n|$)",  # Standalone number
    ]

    response_lower = response.lower().strip()

    for pattern in patterns:
        matches = re.findall(pattern, response_lower, re.MULTILINE | re.IGNORECASE)
        if matches:
            # Get the last match (most likely to be final answer)
            answer = matches[-1]
            try:
                num = int(answer)
                if 0 <= num <= 999:  # AIME answers are in range 0-999
                    return str(num)
            except ValueError:
                continue

    # If no clear pattern found, try to extract any 1-3 digit number
    numbers = re.findall(r"\b(\d{1,3})\b", response)
    if numbers:
        for num_str in reversed(numbers):  # Check from end
            try:
                num = int(num_str)
                if 0 <= num <= 999:
                    return str(num)
            except ValueError:
                continue

    return ""


def get_num_tokens(text, tokenizer_instance):
    """Count tokens in text"""
    if not text:
        return 0
    encoding = tokenizer_instance(text, return_tensors = "pt")
    return len(encoding["input_ids"][0])


def evaluate_model_aime(
    model,
    tokenizer,
    model_type = "base",
    lora_request = None,
    temperature = 0.3,
    n_sampling = 8,
    max_tokens = 32768,
    top_p = 0.95,
    seed = 0,
):
    """Evaluate model on combined AIME dataset with official configuration"""

    print(f"\n{'='*70}")
    print(f"🧮 AIME EVALUATION - {model_type.upper()} MODEL")
    print(f"Combined Dataset: test2024 + test2025-I + test2025-II")
    print(f"{'='*70}")

    # Load combined AIME dataset
    try:
        eval_dataset = load_aime_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    if not eval_dataset:
        print("No examples found in dataset")
        return None

    # Initialize tracking variables
    records = {}
    input_tokens = []
    output_tokens = []
    correct_answers = 0

    # Track performance by source dataset
    source_stats = {}
    for example in eval_dataset:
        source = example["source_dataset"]
        if source not in source_stats:
            source_stats[source] = {"total": 0, "correct": 0}
        source_stats[source]["total"] += 1

    # Setup sampling parameters (AIME configuration)
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = top_p,
        max_tokens = max_tokens,
        n = n_sampling,  # Multiple samples per question
        seed = seed,
    )

    print(f"\n🔧 Configuration:")
    print(f"   Temperature: {temperature}")
    print(f"   Samples per question: {n_sampling}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Top-p: {top_p}")
    print(f"   Seed: {seed}")

    # Temporarily suppress verbose logging
    original_levels = {}
    loggers_to_suppress = [
        "vllm",
        "vllm.engine",
        "vllm.worker",
        "vllm.model_executor",
        "vllm.executor",
        "ray",
    ]

    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)

    try:
        print(f"\n🚀 Evaluating {len(eval_dataset)} problems...")

        # Main evaluation loop
        with tqdm(
            total = len(eval_dataset), desc = "Processing AIME problems", unit = "problem"
        ) as pbar:
            for task_id, item in enumerate(eval_dataset):
                try:
                    # Prepare prompt
                    prompt_text = tokenizer.apply_chat_template(
                        item["prompt"], add_generation_prompt = True, tokenize = False
                    )

                    input_tokens.append(get_num_tokens(prompt_text, tokenizer))

                    # Generate multiple responses
                    outputs = model.fast_generate(
                        [prompt_text],
                        sampling_params = sampling_params,
                        lora_request = lora_request,
                        use_tqdm = False,
                    )[0].outputs

                    # Process all generated responses
                    responses = [output.text for output in outputs]
                    extracted_answers = [
                        extract_aime_answer(response) for response in responses
                    ]

                    # Calculate total output tokens
                    total_output_tokens = sum(
                        get_num_tokens(response, tokenizer) for response in responses
                    )
                    output_tokens.append(total_output_tokens)

                    # Check if any answer is correct
                    ground_truth = item["answer"]
                    correct_responses = [
                        ans == ground_truth for ans in extracted_answers
                    ]
                    is_correct = any(correct_responses)

                    if is_correct:
                        correct_answers += 1
                        source_stats[item["source_dataset"]]["correct"] += 1

                    # Store detailed record
                    records[task_id] = {
                        "global_id": item["global_id"],
                        "original_id": item["original_id"],
                        "source_dataset": item["source_dataset"],
                        "problem": item["problem"],
                        "ground_truth": ground_truth,
                        "responses": responses,
                        "extracted_answers": extracted_answers,
                        "correct_responses": correct_responses,
                        "is_correct": is_correct,
                        "input_tokens": input_tokens[-1],
                        "output_tokens": total_output_tokens,
                        "n_correct": sum(correct_responses),
                        "n_total": len(responses),
                        "solution": item.get("solution", ""),
                        "url": item.get("url", ""),
                    }

                    # Update progress
                    current_accuracy = correct_answers / (task_id + 1) * 100
                    pbar.set_postfix(
                        {
                            "accuracy": f"{current_accuracy:.1f}%",
                            "correct": correct_answers,
                            "total": task_id + 1,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing problem {task_id}: {str(e)}")
                    records[task_id] = {
                        "global_id": item.get("global_id", task_id),
                        "original_id": item.get("original_id", task_id),
                        "source_dataset": item.get("source_dataset", "unknown"),
                        "problem": item["problem"],
                        "ground_truth": item["answer"],
                        "error": str(e),
                        "is_correct": False,
                    }
                    pbar.update(1)
                    continue

    finally:
        # Restore logging levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

    # Calculate metrics
    total_problems = len(eval_dataset)
    accuracy = correct_answers / total_problems * 100

    # Calculate Pass@k (probability that at least one of k samples is correct)
    pass_at_k_scores = []
    for record in records.values():
        if "n_correct" in record and "n_total" in record:
            n_correct = record["n_correct"]
            n_total = record["n_total"]
            if n_correct > 0:
                pass_at_k_scores.append(1.0)
            else:
                pass_at_k_scores.append(0.0)

    pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0

    # Calculate per-source accuracies
    source_accuracies = {}
    for source, stats in source_stats.items():
        source_accuracies[source] = (
            (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )

    results = {
        "model_type": model_type,
        "dataset": "aime_combined",
        "total_problems": total_problems,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k * 100,
        "source_stats": source_stats,
        "source_accuracies": source_accuracies,
        "temperature": temperature,
        "n_sampling": n_sampling,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
        "avg_input_tokens": sum(input_tokens) / len(input_tokens)
        if input_tokens
        else 0,
        "avg_output_tokens": sum(output_tokens) / len(output_tokens)
        if output_tokens
        else 0,
        "max_input_tokens": max(input_tokens) if input_tokens else 0,
        "max_output_tokens": max(output_tokens) if output_tokens else 0,
    }

    # Save results
    filename = f"aime_eval_combined_{model_type}_t{temperature}_n{n_sampling}.json"
    with open(filename, "w", encoding = "utf-8") as f:
        json.dump({"results": results, "records": records}, f, indent = 4)

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print(f"📊 AIME EVALUATION RESULTS - {model_type.upper()}")
    print(f"{'='*70}")

    print(f"\n🎯 Overall Performance:")
    print(f"   Total problems:       {total_problems:>6}")
    print(
        f"   Correct answers:      {correct_answers:>6}/{total_problems} ({accuracy:>5.1f}%)"
    )
    print(f"   Pass@{n_sampling}:              {pass_at_k:>10.1f}%")

    print(f"\n📈 Performance by Dataset:")
    for source, stats in source_stats.items():
        source_acc = source_accuracies[source]
        print(
            f"   {source:>12}: {stats['correct']:>3}/{stats['total']:>3} ({source_acc:>5.1f}%)"
        )

    print(f"\n🔧 Configuration:")
    print(f"   Temperature:          {temperature}")
    print(f"   Samples per problem:  {n_sampling}")
    print(f"   Max tokens:           {max_tokens}")
    print(f"   Top-p:                {top_p}")
    print(f"   Seed:                 {seed}")

    print(f"\n📝 Token Statistics:")
    print(f"   Avg input tokens:     {results['avg_input_tokens']:>10.1f}")
    print(f"   Avg output tokens:    {results['avg_output_tokens']:>10.1f}")
    print(f"   Max input tokens:     {results['max_input_tokens']:>10}")
    print(f"   Max output tokens:    {results['max_output_tokens']:>10}")

    # Performance assessment for AIME
    if accuracy >= 50:
        tier = "🏆 EXCEPTIONAL"
    elif accuracy >= 30:
        tier = "✅ EXCELLENT"
    elif accuracy >= 20:
        tier = "🎯 VERY GOOD"
    elif accuracy >= 10:
        tier = "⚠️  GOOD"
    elif accuracy >= 5:
        tier = "📈 FAIR"
    else:
        tier = "❌ NEEDS IMPROVEMENT"

    print(f"\n🎖️  AIME Performance:     {tier} ({accuracy:.1f}%)")
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"\n{'='*70}")

    return results


# Comparison functions for multiple model results
def compare_aime_results(all_results):
    """Generate comprehensive comparison for AIME evaluation results"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE AIME MODEL COMPARISON")
    print(f"{'='*80}")

    # Main comparison table
    print(
        f"{'Model':<15} {'Accuracy %':<12} {'Pass@K %':<10} {'Correct':<8} {'Total':<8}"
    )
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['model_type']:<15} "
            f"{result['accuracy']:<12.1f} "
            f"{result['pass_at_k']:<10.1f} "
            f"{result['correct_answers']:<8} "
            f"{result['total_problems']:<8}"
        )

    # Performance improvement analysis
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*50}")

        base_result = all_results[0]  # Assume first is base model

        for i, result in enumerate(all_results[1:], 1):
            print(f"\n{result['model_type']} vs {base_result['model_type']}:")

            accuracy_improvement = result["accuracy"] - base_result["accuracy"]
            pass_k_improvement = result["pass_at_k"] - base_result["pass_at_k"]

            print(f"  Accuracy improvement:  {accuracy_improvement:+.1f}%")
            print(f"  Pass@K improvement:    {pass_k_improvement:+.1f}%")

    # Dataset breakdown
    print(f"\n{'='*50}")
    print("PERFORMANCE BY DATASET")
    print(f"{'='*50}")

    # Get all unique datasets from the first result
    if all_results and "source_accuracies" in all_results[0]:
        datasets = list(all_results[0]["source_accuracies"].keys())

        print(f"{'Model':<15}", end = "")
        for dataset in datasets:
            print(f"{dataset:<15}", end = "")
        print()
        print("-" * (15 + 15 * len(datasets)))

        for result in all_results:
            print(f"{result['model_type']:<15}", end = "")
            for dataset in datasets:
                accuracy = result["source_accuracies"].get(dataset, 0)
                print(f"{accuracy:<15.1f}", end = "")
            print()

    # Save comparison
    comparison_data = {
        "summary": all_results,
        "best_model": max(all_results, key = lambda x: x["accuracy"]),
    }

    with open("aime_model_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent = 4)

    print(
        f"\nBest performing model: {comparison_data['best_model']['model_type']} "
        f"({comparison_data['best_model']['accuracy']:.1f}% accuracy)"
    )
