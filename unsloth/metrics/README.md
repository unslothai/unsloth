# Unsloth Metrics Collection

Comprehensive runtime performance metrics collection for Unsloth, inspired by vLLM's metrics system.

## Overview

This module provides detailed tracking of both inference and training metrics, with optional Prometheus-compatible export and HTTP server support. Metrics are collected automatically when enabled, requiring no changes to your existing code.

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

## Features

### Inference Metrics
- **Request tracking**: Total requests, active requests, finish reasons
- **Latency metrics**: End-to-end latency (measured), prefill latency (estimated), decode latency (estimated), time per token
- **Token metrics**: Prompt tokens, generation tokens, tokens per second
- **Throughput**: Real-time throughput calculations

**Note**: Prefill and decode latencies are estimated after generation completes. For more accurate metrics, consider hooking into the generation process itself (e.g., via LogitsProcessor or StoppingCriteria).

### Training Metrics
- **Step tracking**: Total steps, samples processed
- **Performance**: Forward/backward pass times (estimated), samples per second
- **Training state**: Loss, learning rate, gradient norm
- **Batch metrics**: Batch size tracking

**Note**: Forward and backward pass times are estimated as a 40/60 split of the total step duration. For precise timings, more intrusive instrumentation would be required.

### Prometheus Integration
- Prometheus-compatible metrics export
- Standard metric types (Counter, Gauge, Histogram)
- Optional dependency (`prometheus_client`)
- Works without Prometheus (graceful degradation)

### HTTP Server (Optional)
- Background HTTP server for metrics scraping
- Standard `/metrics` endpoint
- Health check endpoint
- Configurable host/port

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

**Counters:**
- `unsloth_request_total` - Total number of requests (labeled by finish_reason)
- `unsloth_prompt_tokens_total` - Total prompt tokens processed
- `unsloth_generation_tokens_total` - Total generation tokens produced

**Gauges:**
- `unsloth_requests_active` - Number of currently active requests
- `unsloth_tokens_per_second` - Current tokens per second throughput

**Histograms:**
- `unsloth_request_latency_seconds` - End-to-end request latency
- `unsloth_prefill_latency_seconds` - Prefill (prompt processing) latency
- `unsloth_decode_latency_seconds` - Decode (generation) latency
- `unsloth_time_per_output_token_seconds` - Time per output token
- `unsloth_prompt_tokens` - Prompt tokens per request
- `unsloth_generation_tokens` - Generation tokens per request

### Training Metrics

**Counters:**
- `unsloth_training_steps_total` - Total training steps
- `unsloth_training_samples_total` - Total samples processed

**Gauges:**
- `unsloth_training_loss` - Current training loss
- `unsloth_learning_rate` - Current learning rate
- `unsloth_training_samples_per_second` - Training throughput
- `unsloth_gradient_norm` - Current gradient norm

**Histograms:**
- `unsloth_training_forward_time_seconds` - Forward pass time
- `unsloth_training_backward_time_seconds` - Backward pass time
- `unsloth_training_batch_size` - Batch size

## How It Works

### Automatic Instrumentation

Metrics are collected automatically when enabled:

1. **Inference**: `unsloth_base_fast_generate()` is instrumented to track:
   - Request lifecycle (start, scheduled, first token, tokens, finish)
   - Latencies (E2E, prefill, decode)
   - Token counts and throughput

2. **Training**: `Trainer.training_step()` is patched to track:
   - Forward/backward pass times
   - Loss, learning rate, gradient norm
   - Batch size and samples per second

### Integration Points

- **Inference hook**: `unsloth/models/vision.py` - `unsloth_base_fast_generate()`
- **Training hook**: `unsloth/models/_utils.py` - `_patch_training_metrics()`
- **Public API**: `unsloth/__init__.py` - Exports metrics functions

### Design Decisions

1. **Non-intrusive**: Metrics are opt-in via `enable_prometheus_metrics()`
2. **Graceful degradation**: Works without Prometheus client installed
3. **Thread-safe**: Singleton pattern with proper locking
4. **Low overhead**: Minimal performance impact when enabled
5. **Modular**: Separate modules for stats, Prometheus, server
6. **vLLM-inspired**: Similar architecture to vLLM's metrics system

## Dependencies

**Optional**: `prometheus_client>=0.20.0` for Prometheus export

Install via:
```bash
pip install prometheus_client
# or
pip install unsloth[metrics]
```

The metrics system works without `prometheus_client` - it gracefully degrades and only provides programmatic access.

## Environment Variables

- `UNSLOTH_ENABLE_METRICS=1` - Enable metrics collection (default: disabled)
- `UNSLOTH_ENABLE_METRICS_TELEMETRY=1` - Enable metrics telemetry (opt-in)
- `UNSLOTH_DISABLE_METRICS_TELEMETRY=1` - Disable metrics telemetry (opt-out)
- `UNSLOTH_METRICS_TELEMETRY_ENDPOINT` - Telemetry endpoint (default: https://api.unsloth.ai/metrics)
- `UNSLOTH_METRICS_TELEMETRY_INTERVAL` - Telemetry interval seconds (default: 300)

## API Reference

### Core Functions

- `enable_prometheus_metrics()` - Enable metrics collection
- `disable_prometheus_metrics()` - Disable metrics collection
- `get_stats_collector()` - Get the global stats collector singleton
- `generate_prometheus_metrics()` - Generate Prometheus-format metrics
- `is_prometheus_available()` - Check if Prometheus client is available

### Telemetry Functions (Server-Side Forwarding)

- `enable_telemetry()` - Enable telemetry (opt-in)
- `disable_telemetry()` - Disable telemetry
- `is_telemetry_enabled()` - Check if telemetry is enabled

### HTTP Server Functions

- `start_metrics_server(host="0.0.0.0", port=9090)` - Start metrics HTTP server
- `stop_metrics_server()` - Stop metrics HTTP server
- `is_metrics_server_running()` - Check if server is running
- `test_metrics_server(port=9090)` - Test server connectivity

### Stats Classes

- `StatsCollector` - Global singleton that manages inference and training stats
- `InferenceStats` - Inference metrics collection
- `TrainingStats` - Training metrics collection

## Examples

### Basic Usage

```python
from unsloth import enable_prometheus_metrics, get_stats_collector

enable_prometheus_metrics()

# Run inference/training
# ...

# Get metrics
stats = get_stats_collector().get_all_stats()
print(f"Inference requests: {stats['inference']['total_requests']}")
print(f"Training steps: {stats['training']['total_steps']}")
```

### Prometheus Export

```python
from unsloth import generate_prometheus_metrics

# Get Prometheus format
metrics_text = generate_prometheus_metrics()

# Print or save
print(metrics_text.decode())
# or
with open("metrics.prom", "wb") as f:
    f.write(metrics_text)
```

### HTTP Server

```python
from unsloth import start_metrics_server, test_metrics_server

# Start server
start_metrics_server(port=9090)

# Test connection
test_metrics_server(port=9090)

# Prometheus can now scrape from http://localhost:9090/metrics
```

## Comparison with vLLM

This implementation is inspired by vLLM's metrics collection:

- Similar metrics structure (request stats, latency breakdowns, token counts)
- Prometheus-compatible export format
- HTTP endpoint for metrics scraping
- Sliding window aggregation for recent metrics

The main difference is that Unsloth's metrics are integrated into the Transformers-based training/inference pipeline rather than a custom engine.

## Testing

Comprehensive test suite available:
```bash
python3 tests/metrics/test_metrics_standalone.py
```

Tests cover:
- Inference metrics tracking
- Training metrics tracking
- StatsCollector singleton pattern
- Prometheus export (when available)

## Future Enhancements

Potential future improvements:
- More detailed latency breakdowns
- Per-model metrics
- Metrics aggregation windows
- Export to other formats (JSON, CSV)
- Metrics dashboard integration
