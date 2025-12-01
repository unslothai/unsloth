"""
OCR Model Evaluation Module

This module provides functionality to evaluate OCR models on datasets with
word error rate (WER) and character error rate (CER) metrics.
"""

import os
import torch
from tqdm import tqdm
import pandas as pd
from jiwer import wer, cer
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import traceback


class OCRModelEvaluator:
    """
    A comprehensive OCR model evaluator that supports multiple models and provides
    detailed analysis with WER and CER metrics.
    """

    def __init__(self):
        """Initialize the OCR evaluator."""
        self.model_comparison_results = {}

    def evaluate_model(
        self,
        model: Any,
        processor: Any,
        dataset: List[Dict],
        output_dir: str = "ocr_evaluation_results",
        max_new_tokens: int = 1024,
        temperature: float = 1.5,
        min_p: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate a model on an OCR dataset.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok = True)

        # Initialize results storage
        results = []

        # Process each sample in the dataset
        for i, sample in enumerate(
            tqdm(dataset, desc = "Evaluating OCR performance", disable = not verbose)
        ):
            try:
                # Extract components from sample
                messages = sample["messages"]

                # Get ground truth, image, and question
                ground_truth, image, question, input_messages = (
                    self._extract_sample_components(messages, i, verbose)
                )

                if ground_truth is None or image is None or question is None:
                    continue

                # Generate model response
                generated_response = self._generate_response(
                    model, processor, input_messages, max_new_tokens, temperature, min_p
                )

                # Calculate metrics
                word_error = wer(ground_truth, generated_response)
                char_error = cer(ground_truth, generated_response)

                # Save individual result
                self._save_individual_result(
                    output_dir,
                    i,
                    question,
                    generated_response,
                    ground_truth,
                    word_error,
                    char_error,
                )

                # Store results for summary
                results.append(
                    {
                        "sample_id": i,
                        "wer": word_error,
                        "cer": char_error,
                        "model_output": generated_response.strip(),
                        "ground_truth": ground_truth,
                        "question": question,
                    }
                )

            except Exception as e:
                if verbose:
                    print(f"Error processing sample {i}: {str(e)}")
                    traceback.print_exc()

        # Generate summary report
        return self._generate_summary_report(results, output_dir, verbose)

    def _extract_sample_components(
        self, messages: List[Dict], sample_idx: int, verbose: bool
    ) -> Tuple[Optional[str], Optional[Any], Optional[str], List[Dict]]:
        """Extract ground truth, image, question, and input messages from sample."""

        # Extract system message (if present)
        system_message = next(
            (msg for msg in messages if msg["role"] == "system"), None
        )

        # Extract user message with the image and question
        user_message = next((msg for msg in messages if msg["role"] == "user"), None)
        if not user_message:
            if verbose:
                print(f"Skipping sample {sample_idx}: No user message found")
            return None, None, None, []

        # Extract assistant message with ground truth
        assistant_message = next(
            (msg for msg in messages if msg["role"] == "assistant"), None
        )
        if not assistant_message:
            if verbose:
                print(
                    f"Skipping sample {sample_idx}: No assistant message (ground truth) found"
                )
            return None, None, None, []

        # Extract ground truth text
        ground_truth = None
        for content_item in assistant_message["content"]:
            if content_item["type"] == "text":
                ground_truth = content_item["text"]
                break

        if not ground_truth:
            if verbose:
                print(
                    f"Skipping sample {sample_idx}: No text found in assistant message"
                )
            return None, None, None, []

        # Extract image and question from user message
        image = None
        question = None

        for content_item in user_message["content"]:
            if content_item["type"] == "image":
                image = content_item["image"]
            elif content_item["type"] == "text":
                question = content_item["text"]

        if not image:
            if verbose:
                print(f"Skipping sample {sample_idx}: No image found in user message")
            return None, None, None, []

        if not question:
            if verbose:
                print(
                    f"Skipping sample {sample_idx}: No question found in user message"
                )
            return None, None, None, []

        # Construct messages for the model input (excluding assistant message)
        input_messages = []
        if system_message:
            input_messages.append(system_message)
        input_messages.append(user_message)

        return ground_truth, image, question, input_messages

    def _generate_response(
        self,
        model: Any,
        processor: Any,
        input_messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        min_p: float,
    ) -> str:
        """Generate response from the model."""

        # Preparation for inference using Qwen's specific processing
        text = processor.apply_chat_template(
            input_messages, tokenize = False, add_generation_prompt = True
        )

        # Process vision info (images/videos) from messages
        image_inputs, video_inputs = process_vision_info(input_messages)

        # Create model inputs
        inputs = processor(
            text = [text],
            images = image_inputs,
            videos = video_inputs,
            padding = True,
            return_tensors = "pt",
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                min_p = min_p,
                use_cache = True,
            )

        # Extract only the generated part (not the input)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated text
        generated_response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = False,
        )[0]

        return generated_response

    def _save_individual_result(
        self,
        output_dir: str,
        sample_idx: int,
        question: str,
        generated_response: str,
        ground_truth: str,
        word_error: float,
        char_error: float,
    ):
        """Save individual sample result to file."""
        output_file = os.path.join(output_dir, f"sample_{sample_idx}.txt")
        with open(output_file, "w", encoding = "utf-8") as f:
            f.write(f"Sample {sample_idx}\n")
            f.write(f"Question: {question}\n\n")
            f.write(f"Model output:\n{generated_response.strip()}\n\n")
            f.write(f"Ground truth:\n{ground_truth}\n\n")
            f.write(f"WER: {word_error:.4f}, CER: {char_error:.4f}")

    def _generate_summary_report(
        self, results: List[Dict], output_dir: str, verbose: bool
    ) -> Tuple[Optional[float], Optional[float]]:
        """Generate and save summary report."""
        if not results:
            if verbose:
                print("No results to summarize.")
            return None, None

        df = pd.DataFrame(results)

        # Calculate overall averages
        avg_wer = df["wer"].mean()
        avg_cer = df["cer"].mean()

        # Save average metrics
        with open(os.path.join(output_dir, "avg_metrics.txt"), "w") as f:
            f.write(f"Average WER: {avg_wer:.4f}\n")
            f.write(f"Average CER: {avg_cer:.4f}\n")

        # Save detailed results
        df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index = False)

        if verbose:
            print("\nResults Summary:")
            print(f"Average WER: {avg_wer:.4f}")
            print(f"Average CER: {avg_cer:.4f}")
            print(f"\nDetailed results saved to {output_dir}/")

        return avg_wer, avg_cer

    def add_to_comparison(self, model_name: str, wer: float, cer: float):
        """Add model results to the comparison tracker."""
        self.model_comparison_results[model_name] = {"wer": wer, "cer": cer}

    def print_model_comparison(
        self, save_csv: bool = True, save_plot: bool = True
    ) -> Optional[pd.DataFrame]:
        """Print a comparison of all models evaluated so far."""
        if not self.model_comparison_results:
            print("No model results available for comparison")
            return None

        print("\n==== MODEL COMPARISON REPORT ====")

        # Create a comparison dataframe
        comparison_df = pd.DataFrame(
            {
                "Model": list(self.model_comparison_results.keys()),
                "WER": [
                    results["wer"] for results in self.model_comparison_results.values()
                ],
                "CER": [
                    results["cer"] for results in self.model_comparison_results.values()
                ],
            }
        )

        # Sort by WER (best performance first)
        comparison_df = comparison_df.sort_values("WER")

        # Display the comparison table
        print("\nComparison Table (sorted by WER):")
        print(comparison_df.to_string(index = False))

        # Save the comparison table
        if save_csv:
            comparison_file = "model_comparison_results.csv"
            comparison_df.to_csv(comparison_file, index = False)
            print(f"\nComparison table saved to {comparison_file}")

        # Generate a bar chart visualization
        if save_plot:
            self._create_comparison_plot(comparison_df)

        return comparison_df

    def _create_comparison_plot(self, comparison_df: pd.DataFrame):
        """Create and save comparison plot."""
        plt.figure(figsize = (12, 6))

        # Plot WER
        plt.subplot(1, 2, 1)
        plt.bar(comparison_df["Model"], comparison_df["WER"], color = "skyblue")
        plt.title("Word Error Rate Comparison")
        plt.ylabel("WER (lower is better)")
        plt.ylim(bottom = 0)
        plt.xticks(rotation = 45, ha = "right")

        # Plot CER
        plt.subplot(1, 2, 2)
        plt.bar(comparison_df["Model"], comparison_df["CER"], color = "lightgreen")
        plt.title("Character Error Rate Comparison")
        plt.ylabel("CER (lower is better)")
        plt.ylim(bottom = 0)
        plt.xticks(rotation = 45, ha = "right")

        plt.tight_layout()
        plt.savefig("ocr_model_comparison.png")
        plt.show()

        print(f"\nVisualization saved to ocr_model_comparison.png")

    def get_comparison_results(self) -> Dict[str, Dict[str, float]]:
        """Get the current comparison results."""
        return self.model_comparison_results.copy()

    def clear_comparison_results(self):
        """Clear all comparison results."""
        self.model_comparison_results.clear()


def evaluate_ocr_model(
    model, processor, dataset, output_dir = "ocr_evaluation_results", **kwargs
):
    """
    Convenience function that maintains backward compatibility with the original function.
    """
    evaluator = OCRModelEvaluator()
    return evaluator.evaluate_model(model, processor, dataset, output_dir, **kwargs)


def create_evaluator():
    """Create a new OCR evaluator instance."""
    return OCRModelEvaluator()
