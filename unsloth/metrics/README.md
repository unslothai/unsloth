# Unsloth Metrics Collection

Comprehensive runtime performance metrics collection for Unsloth, similar to vLLM's metrics system.

## Quick Start

```python
from unsloth import enable_prometheus_metrics, get_stats_collector

# Enable metrics (once at start)
enable_prometheus_metrics()

# Use Unsloth normally - metrics collected automatically
model, tokenizer = FastLanguageModel.from_pretrained(...)
output = model.generate(...)

# Access metrics programmatically
stats = get_stats_collector().get_all_stats()
print("Inference:", stats['inference'])
print("Training:", stats['training'])
```

## Usage

### Programmatic Access (Recommended)

```python
from unsloth import enable_prometheus_metrics, get_stats_collector, generate_prometheus_metrics

enable_prometheus_metrics()

# ... your code ...

# Get stats
stats = get_stats_collector().get_all_stats()
inference = stats['inference']
print(f"Requests: {inference['total_requests']}, Tokens/sec: {inference['tokens_per_second']:.2f}")

# Get Prometheus format
prometheus_text = generate_prometheus_metrics()
print(prometheus_text.decode())
```

### Optional: HTTP Server

```python
from unsloth import start_metrics_server

start_metrics_server(port=9090)
# Access at http://localhost:9090/metrics
```

## Collected Metrics

### Inference Metrics
- Request counts, latencies (E2E, prefill, decode)
- Token counts (prompt, generation)
- Throughput (tokens/sec, time per token)

### Training Metrics
- Training steps, samples processed
- Loss, learning rate, gradient norm
- Throughput (samples/sec)
- Forward/backward pass times

## Dependencies

Optional: `pip install prometheus_client` for Prometheus export

## Environment Variables

- `UNSLOTH_ENABLE_METRICS=1`: Enable metrics collection (default: disabled)
