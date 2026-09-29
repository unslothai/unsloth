# Unsloth runtime metrics

Opt-in inference and training metrics with optional Prometheus export. Off by default; when off,
each `generate()` / `training_step` call pays one attribute read.

```python
from unsloth import enable_prometheus_metrics, get_stats_collector

enable_prometheus_metrics()          # or UNSLOTH_ENABLE_METRICS=1 for in-process stats only
# ... model.generate(...) / trainer.train() as usual ...
stats = get_stats_collector().get_all_stats()
print(stats["inference"], stats["training"])
```

Serve `/metrics` for Prometheus (loopback by default; `host="0.0.0.0"` exposes it on every
interface, `port=0` picks a free port):

```python
from unsloth import start_metrics_server
start_metrics_server(port = 9090)   # http://127.0.0.1:9090/metrics
```

`pip install unsloth[metrics]` (or `prometheus_client`) enables export; without it the
in-process stats still work and `/metrics` returns a comment line.

## What is measured

Inference: every `generate()` of a model loaded by `FastLanguageModel` / `FastModel` /
`FastVisionModel` is one request.

| Metric | Meaning |
|---|---|
| `unsloth_request_total{finish_reason}` | requests; `length` if the longest row hit `max_new_tokens`, `error` if generate raised, else `stop` |
| `unsloth_prompt_tokens_total`, `unsloth_prompt_tokens_per_request` | prompt tokens (token-id inputs only; embeds / audio count 0) |
| `unsloth_generation_tokens_total`, `unsloth_generation_tokens_per_request` | returned tokens minus prompt, summed over rows (padding after an early EOS counts) |
| `unsloth_request_latency_seconds` | wall time of `generate()` |
| `unsloth_time_per_output_token_seconds` | request latency / generated tokens (includes prefill) |
| `unsloth_requests_active`, `unsloth_tokens_per_second` | gauges over the last 1000 requests |

Prefill / decode / time-to-first-token are not reported: measuring them needs a per-token
host sync inside generate.

Training: `Trainer.training_step` (one call per micro-batch) is wrapped.

| Metric | Meaning |
|---|---|
| `unsloth_training_steps_total`, `unsloth_training_samples_total` | micro-batches and sequences (padding-free rows split on `position_ids`) |
| `unsloth_training_loss` | last micro-batch loss x GA; its mean over an accumulation window equals the loss Trainer logs |
| `unsloth_learning_rate` | scheduler's last LR |
| `unsloth_training_step_time_seconds` | wall time of forward + backward (not split) |
| `unsloth_training_samples_per_second`, `unsloth_training_batch_size` | throughput, batch size |

Enabled training metrics cost one `loss.item()` host sync per micro-batch.

## Telemetry (opt-in)

`UNSLOTH_ENABLE_METRICS_TELEMETRY=1` or `enable_telemetry()` POSTs aggregated counts and
averages (never prompts or outputs) at most once per `UNSLOTH_METRICS_TELEMETRY_INTERVAL`
seconds (default 300) to `UNSLOTH_METRICS_TELEMETRY_ENDPOINT`.
`UNSLOTH_DISABLE_METRICS_TELEMETRY=1` always wins. If
`UNSLOTH_METRICS_TELEMETRY_SETTINGS_ENDPOINT` is set it is read once and `{"enabled": false}`
turns telemetry off.
