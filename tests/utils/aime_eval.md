# AIME Dataset Evaluator

A Python module for evaluating language models on the AIME (American Invitational Mathematics Examination) dataset. This evaluator automatically downloads and combines multiple AIME test datasets and provides comprehensive mathematical reasoning assessment.


## Basic Usage

```python
from aime_utils import evaluate_model_aime

# Simple AIME evaluation
results = evaluate_model_aime(
    model=your_model,
    tokenizer=your_tokenizer,
    model_type="base_model",
    temperature=0.3,
    n_sampling=8,
    max_tokens=32768
)

print(f"AIME Accuracy: {results['accuracy']:.1f}%")
print(f"Pass@8: {results['pass_at_k']:.1f}%")
```

## Advanced Usage

```python
from aime_utils import evaluate_model_aime, compare_aime_results

# Evaluate multiple model configurations
all_results = []

# Base model
base_results = evaluate_model_aime(
    model=base_model,
    tokenizer=tokenizer,
    model_type="base",
    temperature=0.3,
    n_sampling=8
)
all_results.append(base_results)

# Fine-tuned model
ft_results = evaluate_model_aime(
    model=finetuned_model,
    tokenizer=tokenizer,
    model_type="finetuned",
    temperature=0.3,
    n_sampling=8
)
all_results.append(ft_results)

# Generate comprehensive comparison
compare_aime_results(all_results)
```

## Dataset Format

The evaluator automatically handles AIME dataset format with problems containing:

- **Problem**: Mathematical question text
- **Answer**: Numerical answer (0-999 range for AIME)
- **Solution**: Step-by-step solution (when available)
- **Source**: Original dataset identifier (test2024, test2025-I, test2025-II)

```python
# Automatic dataset download and formatting
{
    "global_id": 0,
    "original_id": "problem_1",
    "source_dataset": "test2024",
    "problem": "Find the number of...",
    "answer": "123",
    "solution": "Step-by-step solution...",
    "prompt": [
        {"role": "system", "content": "You are a mathematical problem solver..."},
        {"role": "user", "content": "Problem: Find the number of..."}
    ]
}
```


## Configuration Examples

### Conservative Evaluation
```python
# Lower temperature for more consistent answers
results = evaluate_model_aime(
    model=model,
    tokenizer=tokenizer,
    model_type="conservative",
    temperature=0.1,
    n_sampling=4,
    top_p=0.9
)
```

### High-Sample Evaluation
```python
# More samples for better Pass@K estimation
results = evaluate_model_aime(
    model=model,
    tokenizer=tokenizer,
    model_type="high_sample",
    temperature=0.5,
    n_sampling=16,
    max_tokens=16384
)
```

### Memory-Optimized
```python
# Reduced parameters for limited resources
results = evaluate_model_aime(
    model=model,
    tokenizer=tokenizer,
    model_type="lite",
    temperature=0.3,
    n_sampling=4,
    max_tokens=8192
)
```

## Examples

### Complete Model Pipeline Evaluation
```python
from aime_utils import evaluate_model_aime, compare_aime_results

def evaluate_training_pipeline(base_model, finetuned_model, merged_model, tokenizer):
    """Evaluate complete training pipeline on AIME"""

    all_results = []

    # Standard evaluation configuration
    eval_config = {
        "temperature": 0.3,
        "n_sampling": 8,
        "max_tokens": 32768,
        "top_p": 0.95,
        "seed": 0
    }

    # Evaluate base model
    print("Evaluating base model...")
    base_results = evaluate_model_aime(
        model=base_model,
        tokenizer=tokenizer,
        model_type="base",
        **eval_config
    )
    all_results.append(base_results)

    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    ft_results = evaluate_model_aime(
        model=finetuned_model,
        tokenizer=tokenizer,
        model_type="finetuned",
        **eval_config
    )
    all_results.append(ft_results)

    # Evaluate merged model
    print("Evaluating merged model...")
    merged_results = evaluate_model_aime(
        model=merged_model,
        tokenizer=tokenizer,
        model_type="merged",
        **eval_config
    )
    all_results.append(merged_results)

    # Generate comparison report
    compare_aime_results(all_results)

    return all_results
```

### Quantization Impact Analysis
```python
def analyze_quantization_impact(model_paths, tokenizer):
    """Analyze impact of different quantization levels"""

    quantization_configs = {
        "fp16": {"load_in_4bit": False, "load_in_8bit": False},
        "8bit": {"load_in_4bit": False, "load_in_8bit": True},
        "4bit": {"load_in_4bit": True, "load_in_8bit": False}
    }

    all_results = []

    for quant_name, load_config in quantization_configs.items():
        print(f"Evaluating {quant_name} quantization...")

        # Load model with specific quantization
        model = load_model_with_config(model_paths["merged"], **load_config)

        results = evaluate_model_aime(
            model=model,
            tokenizer=tokenizer,
            model_type=f"merged_{quant_name}",
            temperature=0.3,
            n_sampling=8,
            max_tokens=32768
        )
        all_results.append(results)

        # Cleanup
        del model
        torch.cuda.empty_cache()

    compare_aime_results(all_results)
    return all_results
```

## Output Format

### Individual Evaluation Results
```
🧮 AIME EVALUATION - BASE MODEL
Combined Dataset: test2024 + test2025-I + test2025-II
====================================================================

🎯 Overall Performance:
   Total problems:           45
   Correct answers:         12/45 (26.7%)
   Pass@8:                  31.1%

📈 Performance by Dataset:
    test2024:   4/15 (26.7%)
  test2025-I:   5/15 (33.3%)
 test2025-II:   3/15 (20.0%)

🎖️  AIME Performance:     ✅ EXCELLENT (26.7%)
```

### Comparison Report
```
COMPREHENSIVE AIME MODEL COMPARISON
================================================================================
Model           Accuracy %   Pass@K %   Correct  Total
--------------------------------------------------------------------------------
finetuned       31.1         35.6       14       45
base            26.7         31.1       12       45
merged_4bit     24.4         28.9       11       45

IMPROVEMENT ANALYSIS
==================================================
finetuned vs base:
  Accuracy improvement:  +4.4%
  Pass@K improvement:    +4.5%
```

## Performance Tiers

The evaluator provides performance assessment based on AIME difficulty:

- **🏆 EXCEPTIONAL**: ≥50% accuracy
- **✅ EXCELLENT**: ≥30% accuracy
- **🎯 VERY GOOD**: ≥20% accuracy
- **⚠️ GOOD**: ≥10% accuracy
- **📈 FAIR**: ≥5% accuracy
- **❌ NEEDS IMPROVEMENT**: <5% accuracy
