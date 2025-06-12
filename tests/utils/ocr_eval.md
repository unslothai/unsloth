
# OCR Model Evaluator
A comprehensive Python module for evaluating Optical Character Recognition (OCR) models using Word Error Rate (WER) and Character Error Rate (CER) metrics. This evaluator supports vision-language models and provides detailed analysis with comparison capabilities across multiple models

## Basic Usage

```python
from ocr_evaluator import evaluate_ocr_model

# Simple evaluation
avg_wer, avg_cer = evaluate_ocr_model(
    model=your_model,
    processor=your_processor,
    dataset=your_dataset,
    output_dir="evaluation_results"
)

print(f"Average WER: {avg_wer:.4f}")
print(f"Average CER: {avg_cer:.4f}")
```


### Dataset Format

The evaluator expects datasets in a chatml conversational format with the following structure:
```
dataset = [
    {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an OCR system."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from this image"},
                    {"type": "image", "image": PIL_Image_object}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Ground truth text"}]
            }
        ]
    },
    # ... more samples
]
```


## Examples

### Document OCR evaluation

```python
from ocr_evaluator import OCRModelEvaluator
from datasets import load_dataset

# Load document OCR dataset
dataset = load_dataset("your-ocr-dataset", split="test")

# Convert to required format
eval_data = [format_document_sample(sample) for sample in dataset]

# Evaluate models
evaluator = OCRModelEvaluator()

# Compare different model configurations
configs = {
    "Standard Model": {"temperature": 1.0, "max_new_tokens": 512},
    "Conservative Model": {"temperature": 0.7, "max_new_tokens": 256},
    "Creative Model": {"temperature": 1.5, "max_new_tokens": 1024}
}

for config_name, params in configs.items():
    wer, cer = evaluator.evaluate_model(
        model=base_model,
        processor=processor,
        dataset=eval_data,
        output_dir=f"document_ocr_{config_name.lower().replace(' ', '_')}",
        **params
    )
    evaluator.add_to_comparison(config_name, wer, cer)

# Generate final report
evaluator.print_model_comparison()
```

### Handwriting Recognition
```python
# Specialized evaluation for handwriting
def evaluate_handwriting_models(models, handwriting_dataset):
    evaluator = OCRModelEvaluator()

    for model_name, (model, processor) in models.items():
        # Adjust parameters for handwriting recognition
        wer, cer = evaluator.evaluate_model(
            model=model,
            processor=processor,
            dataset=handwriting_dataset,
            temperature=1.2,  # Slightly higher for handwriting variety
            max_new_tokens=128,  # Usually shorter text
            output_dir=f"handwriting_{model_name}"
        )
        evaluator.add_to_comparison(f"Handwriting - {model_name}", wer, cer)

    return evaluator.print_model_comparison()
```
