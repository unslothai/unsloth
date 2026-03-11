# -*- coding: utf-8 -*-
"""test_Llama3_1_(3B)_GRPO_LoRA (1).ipynb

### Unsloth

"""

from unsloth import FastLanguageModel
import torch
import sys
from pathlib import Path
import multiprocessing as mp
import gc
from multiprocessing import Queue

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.aime_eval import evaluate_model_aime, compare_aime_results


max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower


def evaluate_merged_model(result_queue, load_in_4bit = False, load_in_8bit = False):
    from unsloth import FastLanguageModel
    from tests.utils.aime_eval import evaluate_model_aime

    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./final_merged_model",
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # False for LoRA 16bit
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.8,  # Reduce if out of memory
    )

    print(f"\n{'='*60}")
    if load_in_4bit:
        print("🔍 EVALUATION Merged model: 4 bits load")
        model_type = "merged_model_4bits"
    elif load_in_8bit:
        print("🔍 EVALUATION Merged model: 8 bits load")
        model_type = "merged_model_8bits"
    else:
        print("🔍 EVALUATION Merged model: 16 bits load")
        model_type = "merged_model_16bits"
    print(f"{'='*60}")

    evaluate_model_aime(
        model = model,
        tokenizer = tokenizer,
        model_type = model_type,
        temperature = 0.3,
        n_sampling = 8,
        max_tokens = 32768,
        top_p = 0.95,
        seed = 0,
    )

    result_queue.put(results)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# Main execution code should be wrapped in this guard
def training_run(result_queue):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = False,  # False for LoRA 16bit
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.8,  # Reduce if out of memory
    )

    """### Helper Functions
    <a name="Data"></a>

#### Helper functions - Data Prep
    """

    import re
    import json

    reasoning_start = "<reasoning>"
    reasoning_end = "</reasoning>"
    solution_start = "<answer>"
    solution_end = "</answer>"

    def extract_hash_answer(text):
        """Extract answer from GSM8K format"""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def prepare_gsm8k_dataset(dataset):
        """Format GSM8K dataset for training"""
        reasoning_start = "<reasoning>"
        reasoning_end = "</reasoning>"
        solution_start = "<answer>"
        solution_end = "</answer>"

        system_prompt = (
            f"You are given a problem. Think about the problem and reason step by step. "
            f"Place your thinking process between {reasoning_start} and {reasoning_end}. "
            f"Then, provide your final numerical solution between {solution_start}{solution_end}"
        )

        def format_gsm8k(example):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                ],
                "answer": extract_hash_answer(example["answer"]),
            }

        return dataset.map(format_gsm8k)

    def prepare_limo_dataset(dataset):
        """Format LIMO dataset for SFT training"""
        if dataset is None:
            return None

        system_prompt = """You are a helpful reasoning assistant. When given a problem, think through it step by step and provide your answer in the following format:

    <reasoning>
    [Your detailed step-by-step reasoning and solution process]
    </reasoning>
    <answer>
    [Your final numerical answer]
    </answer>"""

        def format_limo(example):
            # Create the assistant response
            assistant_response = f"<reasoning>\n{example['solution']}\n</reasoning>\n<answer>\n{example['answer']}\n</answer>"

            # Return a DICTIONARY with the conversation in a field
            return {
                "prompt": [  # ← This is the key change - wrap in a dict
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        return dataset.map(format_limo)

    print("\n✅ Dataset preparation functions defined!")

    """#### Helper functions - Evaluation"""

    def get_max_prompt_length(dataset, tokenizer):
        """Calculate maximum and average prompt length in dataset"""
        print("Analyzing prompt lengths...")

        lengths = dataset.map(
            lambda x: {
                "tokens": tokenizer.apply_chat_template(
                    x["prompt"], add_generation_prompt = True, tokenize = True
                )
            },
            batched = True,
        ).map(lambda x: {"length": len(x["tokens"])})["length"]

        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)

        print(
            f"Prompt lengths - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f}"
        )
        return max_length, avg_length

    def extract_unsloth_answer(text, start_tag = "<SOLUTION>", end_tag = "</SOLUTION>"):
        """Extract answer from Unsloth SOLUTION tags"""
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            answer = matches[-1]  # Get the last match
            answer = re.sub(r"[%$,]", "", answer).strip()
            return answer
        return ""

    def find_number(search_string):
        """Find the last number in a string"""
        numbers = re.compile(
            r"-?[\d,]*\.?\d+",
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        ).findall(search_string)

        if numbers:
            return numbers[-1].replace(",", "").strip()
        return ""

    def remove_symbols(x: str) -> str:
        """Remove commas, percent and dollar symbols"""
        if not x:
            return ""
        return x.replace(",", "").replace("%", "").replace("$", "").strip()

    def get_num_tokens(text, tokenizer_instance):
        """Count tokens in text"""
        if not text:
            return 0
        encoding = tokenizer_instance(text, return_tensors = "pt")
        return len(encoding["input_ids"][0])

    def check_format_compliance(text, format_type = "unsloth"):
        """Check if response follows expected format"""
        if format_type == "unsloth":
            reasoning_start = "<start_reasoning>"
            reasoning_end = "<end_reasoning>"
            solution_start = "<SOLUTION>"
            solution_end = "</SOLUTION>"

            pattern = (
                rf"^[\s]*{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
                rf"{re.escape(solution_start)}.+?{re.escape(solution_end)}[\s]*$"
            )
        else:
            return False

        return bool(re.match(pattern, text.strip(), re.DOTALL))

    def normalize_answer(answer):
        """Normalize answer for comparison"""
        if not answer:
            return ""

        normalized = remove_symbols(str(answer))

        try:
            float_val = float(normalized)
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        except (ValueError, TypeError):
            return normalized

    def evaluate_answer_correctness(extracted_answer, ground_truth):
        """Evaluate answer correctness with multiple criteria"""
        if not extracted_answer or not ground_truth:
            return False, False, 0.0

        norm_extracted = normalize_answer(extracted_answer)
        norm_ground_truth = normalize_answer(ground_truth)

        if norm_extracted == norm_ground_truth:
            return True, True, 1.0

        try:
            extracted_num = float(norm_extracted)
            ground_truth_num = float(norm_ground_truth)

            if ground_truth_num != 0:
                relative_error = abs(extracted_num - ground_truth_num) / abs(
                    ground_truth_num
                )

                if relative_error < 0.01:
                    return True, True, 0.9
                elif relative_error < 0.05:
                    return False, True, 0.7
                elif relative_error < 0.10:
                    return False, True, 0.5
            else:
                if extracted_num == 0:
                    return True, True, 1.0
                elif abs(extracted_num) < 0.01:
                    return False, True, 0.7

        except (ValueError, TypeError):
            if norm_extracted.lower() == norm_ground_truth.lower():
                return True, True, 1.0

        return False, False, 0.0

    """#### Reward Functions for GRPO"""

    def match_format_exactly(completions, **kwargs):
        """Reward function for exact format matching"""
        reasoning_start = "<reasoning>"
        reasoning_end = "</reasoning>"
        solution_start = "<answer>"
        solution_end = "</answer>"

        pattern = (
            rf"^[\s]*{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
            rf"{re.escape(solution_start)}.+?{re.escape(solution_end)}[\s]*$"
        )

        responses = [completion[0]["content"] for completion in completions]
        rewards = [
            3.0 if re.match(pattern, response, re.DOTALL) else 0.0
            for response in responses
        ]
        return rewards

    def match_format_approximately(completions, **kwargs):
        """Reward function for approximate format matching"""
        reasoning_start = "<reasoning>"
        reasoning_end = "</reasoning>"
        solution_start = "<answerr>"
        solution_end = "</answer>"

        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer_correctness(prompts, completions, answer, **kwargs):
        """Reward function for answer correctness"""

        def extract_solution_answer(text):
            pattern = r"<answer>(.*?)</answer>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return re.sub(r"[%$,]", "", match.group(1)).strip()
            return ""

        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_solution_answer(r) for r in responses]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if not guess:
                scores.append(0)
                continue

            if guess == true_answer:
                score += 3.0
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 1.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 0.5
                    else:
                        score -= 1.5
                except:
                    score -= 1.5
            scores.append(score)
        return scores

    print("✅ Reward functions defined!")

    """#### Main Evaluation Function"""

    import gc

    """#### Comparison and Memory Management"""

    def compare_model_results(all_results):
        """Generate comprehensive comparison of multiple model results"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MODEL COMPARISON")
        print(f"{'='*80}")

        # Main table
        print(
            f"{'Model':<15} {'Format %':<10} {'Exact %':<10} {'Plausible %':<12} {'Confidence':<12}"
        )
        print("-" * 80)

        for result in all_results:
            print(
                f"{result['model_type']:<15} "
                f"{result['correct_format_pct']:<10.1f} "
                f"{result['exact_match_pct']:<10.1f} "
                f"{result['plausible_match_pct']:<12.1f} "
                f"{result['avg_confidence']:<12.3f}"
            )

        # Improvement analysis
        if len(all_results) > 1:
            print(f"\n{'='*50}")
            print("IMPROVEMENT ANALYSIS")
            print(f"{'='*50}")

            base_result = all_results[0]
            for result in all_results[1:]:
                print(f"\n{result['model_type']} vs {base_result['model_type']}:")
                format_improvement = (
                    result["correct_format_pct"] - base_result["correct_format_pct"]
                )
                exact_improvement = (
                    result["exact_match_pct"] - base_result["exact_match_pct"]
                )
                plausible_improvement = (
                    result["plausible_match_pct"] - base_result["plausible_match_pct"]
                )

                print(f"  Format compliance: {format_improvement:+.1f}%")
                print(f"  Exact matches:     {exact_improvement:+.1f}%")
                print(f"  Plausible matches: {plausible_improvement:+.1f}%")

        # Save comparison
        comparison_data = {
            "summary": all_results,
            "best_model": max(all_results, key = lambda x: x["exact_match_pct"]),
        }

        with open("model_comparison_comprehensive.json", "w") as f:
            json.dump(comparison_data, f, indent = 4)

        print(
            f"\nBest performing model: {comparison_data['best_model']['model_type']} "
            f"({comparison_data['best_model']['exact_match_pct']:.1f}% exact matches)"
        )

    def cleanup_memory():
        """Comprehensive memory cleanup"""
        print("🧹 Cleaning up GPU memory...")
        for _ in range(10):
            torch.cuda.empty_cache()
            gc.collect()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )

    """#### Data Loading and Preparation"""

    from datasets import load_dataset

    # Load GSM8K
    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split = "train")

    # Load LIMO (adjust this based on your access method)
    limo_train = load_dataset("GAIR/LIMO", split = "train")

    # Prepare datasets
    gsm8k_train = prepare_gsm8k_dataset(gsm8k_dataset)
    limo_train = prepare_limo_dataset(limo_train)

    print(f"  GSM8K train: {len(gsm8k_train)}")
    print(f"  LIMO train:  {len(limo_train) if limo_train else 0}")

    # Store results
    all_results = []

    # Single temperature evaluation on combined dataset
    results = evaluate_model_aime(
        model = model,
        tokenizer = tokenizer,
        model_type = "base",
        temperature = 0.3,
        n_sampling = 8,
        max_tokens = 32768,
        top_p = 0.95,
        seed = 0,
    )

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["prompt"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize = False, add_generation_prompt = False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    limo_train = limo_train.map(
        formatting_prompts_func,
        batched = True,
    )

    from trl import SFTTrainer
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    from unsloth import is_bfloat16_supported

    print(f"\n{'*'*60}")
    print("🎯 STAGE 1: Qlora Fine-Tuning on LIMO")
    print(f"{'*'*60}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",  # Enable long context finetuning
        random_state = 3407,
    )

    if limo_train is not None:
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = limo_train,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
            dataset_num_proc = 2,
            packing = False,  # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs = 1,  # Set this for 1 full training run.
                # max_steps = 60,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",  # Use this for WandB etc
            ),
        )

        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        # Train
        print(f"🚂 Starting SFT training on {len(limo_train)} examples...")
        trainer.train()

        # Save checkpoint
        model.save_pretrained("qlora_checkpoint")
        tokenizer.save_pretrained("qlora_checkpoint")
        print("💾 Qlora checkpoint saved!")

        # Cleanup
        del trainer
        cleanup_memory()

        print("✅ Qlora training completed!")
    else:
        print("⚠️ Skipping Qlora training - no LIMO dataset available")

    # Cleanup
    cleanup_memory()

    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5

    match_numbers = re.compile(
        solution_start + r".*?([\d\.\,]{1,})", flags = re.MULTILINE | re.DOTALL
    )

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        # Print only every few steps
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print(
                "*" * 20,
                f"Question:\n{question}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted_responses[0]}",
            )
        PRINTED_TIMES += 1

        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                # Remove commas like in 123,456
                guess = float(guess.strip().replace(",", ""))
                scores.append(1.5 if guess == true_answer else -0.5)
            except:
                scores.append(0)
                continue
        return scores

    print(f"\n{'*'*60}")
    print("🎯 STAGE 2: GRPO Fine-Tuning on GSM8K")
    print(f"{'*'*60}")

    # Get max prompt length
    max_prompt_length, _ = get_max_prompt_length(gsm8k_train, tokenizer)
    max_prompt_length = min(max_prompt_length + 10, 512)  # Add buffer, cap at 512

    print(f"Using max_prompt_length: {max_prompt_length}")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate = 5e-6,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch_fused",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,  # Increase to 4 for smoother training
        num_generations = 8,  # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 250,
        max_steps = 1000,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "none",  # Can use Weights & Biases
        output_dir = "outputs",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer_correctness,
            check_numbers,
        ],
        args = training_args,
        train_dataset = gsm8k_train,
    )

    # Train
    print(f"🚂 Starting GRPO training on {len(gsm8k_train)} examples...")
    trainer.train()

    # Save checkpoint
    model.save_pretrained("grpo_checkpoint")
    tokenizer.save_pretrained("grpo_checkpoint")
    print("💾 GRPO checkpoint saved!")

    # Cleanup
    del trainer
    del training_args
    cleanup_memory()

    print("✅ GRPO training completed!")

    print(f"\n{'='*60}")
    print("🔍 EVALUATION 3: Final GRPO Model")
    print(f"{'='*60}")

    grpo_results = evaluate_model_aime(
        model = model,
        tokenizer = tokenizer,
        model_type = "grpo",
        temperature = 0.3,
        n_sampling = 8,
        max_tokens = 32768,
        top_p = 0.95,
        seed = 0,
    )

    all_results.append(grpo_results)
    print("✅ Final model evaluation complete!")

    print(f"\n{'='*60}")
    print("💾 SAVING FINAL MODEL")
    print(f"{'='*60}")

    # Save as merged model
    try:
        model.save_pretrained_merged(
            "final_merged_model", tokenizer, save_method = "merged_16bit"
        )
        print("✅ Merged model saved to: final_merged_model/")
    except Exception as e:
        print(f"⚠️ Could not save merged model: {e}")
        print("Final model saved as LoRA adapter only")

    print("💾 Model saving complete!")

    safe_remove_directory("./unsloth_compiled_cache")

    result_queue.put(results)

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # # Merged model load 16 bits model AIME eval
    # result_queue = mp.Queue()
    # p = mp.Process(target=evaluate_merged_model, args=(result_queue, False, False))
    # p.start()
    # p.join()
    #
    # merged_16bits = result_queue.get()
    # all_results.append(merged_16bits)
    #
    # # Clean up
    # del merged_model
    # del merged_tokenizer
    # del dataset_ppl
    # torch.cuda.empty_cache()
    # gc.collect()
    #
    # safe_remove_directory("./unsloth_compiled_cache")
    #
    # # Merged model load 8 bits model AIME eval
    #
    # result_queue = mp.Queue()
    # p = mp.Process(target=evaluate_merged_model, args=(result_queue, False, True))
    # p.start()
    # p.join()
    #
    # merged_16bits = result_queue.get()
    # all_results.append(merged_16bits)

    # Merged model load 4 bits AIME eval
    # result_queue = mp.Queue()
    # p = mp.Process(target=evaluate_merged_model, args=(result_queue, True, False))
    # p.start()
    # p.join()
    #
    # merged_16bits = result_queue.get()
    # all_results.append(merged_16bits)


if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)
    result_queue = mp.Queue()
    all_results = []

    # run main finetuning and grpo loop
    p = mp.Process(target = training_run, args = (result_queue,))
    p.start()
    p.join()

    results = result_queue.get()
    all_results = results

    # evaluate merged model loaded 16bits
    p = mp.Process(target = evaluate_merged_model, args = (result_queue, False, False))
    p.start()
    p.join()

    merged_load_16bits = result_queue.get()
    all_results.append(merged_load_16bits)
    safe_remove_directory("./unsloth_compiled_cache")

    # Merged model load 8 bits model AIME eval
    p = mp.Process(target = evaluate_merged_model, args = (result_queue, False, True))
    p.start()
    p.join()

    merged_load_8bits = result_queue.get()
    all_results.append(merged_load_8bits)

    safe_remove_directory("./unsloth_compiled_cache")

    # Merged model load 4 bits model AIME eval
    p = mp.Process(target = evaluate_merged_model, args = (result_queue, True, False))
    p.start()
    p.join()

    merged_load_4bits = result_queue.get()
    all_results.append(merged_load_4bits)

    safe_remove_directory("./unsloth_compiled_cache")

    # AIME-specific comparison function

    print(f"\n{'='*80}")
    print("🏆 FINAL TRAINING PIPELINE RESULTS")
    print(f"{'='*80}")

    # Use the AIME-specific comparison
    compare_aime_results(all_results)
