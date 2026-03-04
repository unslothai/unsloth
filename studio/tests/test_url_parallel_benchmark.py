"""
Benchmark: parallel fsspec URL image downloads with ThreadPoolExecutor.
Tests different worker counts to find optimal parallelism.
Dataset: google-research-datasets/conceptual_captions (subset: labeled)
"""
from datasets import load_dataset, Dataset
from PIL import Image as PILImage
from io import BytesIO
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
import fsspec
import time
import os

DATASET = "google-research-datasets/conceptual_captions"
SUBSET = "labeled"
SPLIT = "train"
N_SAMPLES = 500

# safe_num_proc formula from studio/backend/utils/hardware/hardware.py
cpu_count = os.cpu_count()
safe_workers = max(1, cpu_count // 3)
print(f"CPU count: {cpu_count}, safe_num_proc: {safe_workers}")

WORKER_COUNTS = [1, 4, 8, 16, 32, safe_workers]
# Deduplicate and sort
WORKER_COUNTS = sorted(set(WORKER_COUNTS))

print(f"Loading {N_SAMPLES} samples from {DATASET} (streaming)...")
ds = load_dataset(DATASET, name=SUBSET, split=SPLIT, streaming=True)
rows = list(islice(ds, N_SAMPLES))
dataset = Dataset.from_list(rows)
urls = [row["image_url"] for row in dataset]
print(f"Loaded {len(urls)} URLs")
print()


def download_single(url):
    """Download a single image URL using fsspec. Returns PIL image or raises."""
    with fsspec.open(url, "rb", expand=True) as f:
        img = PILImage.open(BytesIO(f.read())).convert("RGB")
    return img


print(f"{'Workers':>8} | {'Time':>8} | {'OK':>6} | {'Fail':>6} | {'Fail%':>6} | {'img/s':>7} | {'Speedup':>8}")
print("-" * 70)

baseline_throughput = None

for n_workers in WORKER_COUNTS:
    success, fail = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(download_single, url): url for url in urls}
        for future in as_completed(futures):
            try:
                img = future.result(timeout=30)
                success += 1
            except Exception:
                fail += 1

    elapsed = time.time() - t0
    fail_pct = (fail / N_SAMPLES) * 100
    throughput = success / elapsed if elapsed > 0 else 0

    if baseline_throughput is None:
        baseline_throughput = throughput
    speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0

    label = f"{n_workers}"
    if n_workers == safe_workers:
        label += "*"  # mark the safe_num_proc value

    print(f"{label:>8} | {elapsed:>7.1f}s | {success:>6} | {fail:>6} | {fail_pct:>5.1f}% | {throughput:>6.1f}/s | {speedup:>7.1f}x")

print()
print("* = safe_num_proc value")
print("Done.")
