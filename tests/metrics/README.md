# Metrics Tests

## Running Tests

```bash
python3 tests/metrics/test_metrics_standalone.py
```

## Test Coverage

The test suite verifies:

1. **InferenceStats** - Request tracking, token counts, latencies
2. **TrainingStats** - Batch tracking, loss, throughput
3. **StatsCollector** - Singleton pattern, enable/disable
4. **Prometheus Export** - Client availability check

## Test Results

See `TEST_RESULTS.txt` for the latest test run output.

## Expected Output

```
============================================================
Unsloth Metrics Collection - Standalone Test
============================================================

1. Testing InferenceStats...
   ✅ Total requests: 1
   ✅ Prompt tokens: 10
   ✅ Generation tokens: 5
   ✅ Avg latency: 0.087s

2. Testing TrainingStats...
   ✅ Total steps: 3
   ✅ Total samples: 12
   ✅ Avg loss: 0.4500

3. Testing StatsCollector...
   ✅ StatsCollector working

4. Testing Prometheus export...
   ✅ Prometheus client available
   ℹ️  Full Prometheus export test requires GPU environment

============================================================
✅ All metrics tests passed!
============================================================
```
