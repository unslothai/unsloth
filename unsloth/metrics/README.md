# Unsloth Metrics Collection

This module provides comprehensive runtime performance metrics collection for Unsloth, similar to vLLM's metrics system.

## Features

- **Inference Metrics**: Track request latencies, token counts, throughput, and more
- **Training Metrics**: Monitor training steps, loss, learning rate, and throughput
- **Prometheus-Compatible**: Export metrics in Prometheus format for integration with monitoring tools
- **Optional HTTP Server**: Expose metrics via HTTP endpoint (similar to vLLM's `/metrics` endpoint)

## Usage

### Basic Usage

```python
from unsloth import FastLanguageModel, enable_prometheus_metrics, get_stats_collector

# Enable metrics collection
enable_prometheus_metrics()

# Your normal Unsloth code
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Metrics are automatically collected during inference
output = model.generate(...)

# Get statistics
stats = get_stats_collector().get_all_stats()
print(stats)
```

### Expose Metrics via HTTP Server

```python
from unsloth import start_metrics_server

# Start metrics server on port 9090 (default)
start_metrics_server(host="0.0.0.0", port=9090)

# Access metrics at http://localhost:9090/metrics
# Prometheus can scrape from this endpoint
```

### Access Metrics Programmatically

```python
from unsloth import get_stats_collector, generate_prometheus_metrics

# Get current stats
collector = get_stats_collector()
stats = collector.get_all_stats()
print(f"Inference stats: {stats['inference']}")
print(f"Training stats: {stats['training']}")

# Generate Prometheus format metrics
prometheus_text = generate_prometheus_metrics()
print(prometheus_text.decode())
```

## Collected Metrics

### Inference Metrics

- `unsloth_request_total`: Total number of requests (labeled by finish_reason)
- `unsloth_requests_active`: Number of currently active requests
- `unsloth_prompt_tokens_total`: Total prompt tokens processed
- `unsloth_generation_tokens_total`: Total generation tokens produced
- `unsloth_tokens_per_second`: Current throughput
- `unsloth_request_latency_seconds`: End-to-end request latency (histogram)
- `unsloth_prefill_latency_seconds`: Prefill latency (histogram)
- `unsloth_decode_latency_seconds`: Decode latency (histogram)
- `unsloth_time_per_output_token_seconds`: Time per output token (histogram)
- `unsloth_prompt_tokens`: Prompt tokens per request (histogram)
- `unsloth_generation_tokens`: Generation tokens per request (histogram)

### Training Metrics

- `unsloth_training_steps_total`: Total training steps
- `unsloth_training_samples_total`: Total samples processed
- `unsloth_training_loss`: Current training loss
- `unsloth_learning_rate`: Current learning rate
- `unsloth_training_samples_per_second`: Training throughput
- `unsloth_gradient_norm`: Current gradient norm
- `unsloth_training_forward_time_seconds`: Forward pass time (histogram)
- `unsloth_training_backward_time_seconds`: Backward pass time (histogram)
- `unsloth_training_batch_size`: Batch size (histogram)

## Dependencies

The metrics system works without external dependencies, but for Prometheus export you need:

```bash
pip install prometheus_client
```

## Environment Variables

- `UNSLOTH_ENABLE_METRICS=1`: Enable metrics collection (default: disabled)
- `UNSLOTH_DISABLE_STATISTICS=1`: Disable usage statistics (different from metrics)

## Comparison with vLLM

This implementation is inspired by vLLM's metrics collection:

- Similar metrics structure (request stats, latency breakdowns, token counts)
- Prometheus-compatible export format
- HTTP endpoint for metrics scraping
- Sliding window aggregation for recent metrics

The main difference is that Unsloth's metrics are integrated into the Transformers-based training/inference pipeline rather than a custom engine.
