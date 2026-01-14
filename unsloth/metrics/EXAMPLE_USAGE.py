"""
Example usage of Unsloth Metrics Collection

This demonstrates how to use the metrics system for both inference and training.
"""

# Example 1: Basic Inference Metrics
from unsloth import FastLanguageModel, enable_prometheus_metrics, get_stats_collector

# Enable metrics collection
enable_prometheus_metrics()

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Run inference - metrics are automatically collected
prompts = ["Hello, how are you?", "What is machine learning?"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7)

# Get statistics
stats = get_stats_collector().get_all_stats()
print("Inference Stats:", stats["inference"])
print("Training Stats:", stats["training"])


# Example 2: Start HTTP Server for Prometheus Scraping
from unsloth import start_metrics_server

# Start metrics server on port 9090
start_metrics_server(host="0.0.0.0", port=9090)

# Now Prometheus can scrape metrics from http://localhost:9090/metrics
# Or view in browser: http://localhost:9090/metrics


# Example 3: Training with Metrics
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Enable metrics (already enabled above, but shown here for completeness)
enable_prometheus_metrics()

# Load model and dataset
model, tokenizer = FastLanguageModel.from_pretrained(...)
dataset = load_dataset("dataset_name", split="train")

# Create trainer - metrics are automatically collected during training
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
    ),
)

# Train - metrics are collected automatically
trainer.train()

# Get training statistics
stats = get_stats_collector().training_stats.get_stats()
print("Training Stats:", stats)
print(f"  Total steps: {stats['total_steps']}")
print(f"  Average loss: {stats['avg_loss']:.4f}")
print(f"  Samples/sec: {stats['samples_per_second']:.2f}")


# Example 4: Programmatic Access to Metrics
from unsloth import generate_prometheus_metrics

# Generate Prometheus-format metrics string
prometheus_text = generate_prometheus_metrics()
print(prometheus_text.decode())


# Example 5: Check if Metrics are Enabled
from unsloth import get_stats_collector

collector = get_stats_collector()
if collector.is_enabled():
    print("Metrics collection is enabled")
    all_stats = collector.get_all_stats()
    print(all_stats)
else:
    print("Metrics collection is disabled")
    collector.enable()  # Enable it


# Example 6: Reset Metrics
collector = get_stats_collector()
collector.reset_all()  # Reset all statistics
