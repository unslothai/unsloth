# Language Model Perplexity Evaluator

A Python module for evaluating language models using perplexity metrics with sliding window approach for long sequences. This evaluator provides efficient computation of perplexity scores across datasets with model comparison capabilities.

## Basic Usage

```python
from perplexity_evaluator import ppl_model, add_to_comparison, print_model_comparison

# Simple perplexity evaluation
dataset = {"text": ["Your text samples here...", "Another text sample..."]}
perplexity = ppl_model(model, tokenizer, dataset)

print(f"Model Perplexity: {perplexity:.4f}")

# Add to comparison tracker
add_to_comparison("My Model", perplexity)
print_model_comparison()
```

