"""
Benchmark: fsspec URL image download throughput at different dataset sizes.
Dataset: google-research-datasets/conceptual_captions (subset: labeled)

Tests sizes: 100, 200, 300, 500, 1000, 1500, 2000
Reports: time, success/fail rate, throughput (images/sec)
"""
from datasets import load_dataset, Dataset
from PIL import Image as PILImage
from io import BytesIO
from itertools import islice
import fsspec
import time

DATASET = "google-research-datasets/conceptual_captions"
SUBSET = "labeled"
SPLIT = "train"
SIZES = [100, 200, 300, 500, 1000, 1500, 2000]

# Load the max we need in one go
max_size = max(SIZES)
print(f"Loading {max_size} samples from {DATASET} (streaming)...")
ds = load_dataset(DATASET, name=SUBSET, split=SPLIT, streaming=True)
rows = list(islice(ds, max_size))
full_dataset = Dataset.from_list(rows)
print(f"Loaded {len(full_dataset)} samples")
print(f"Columns: {full_dataset.column_names}")
print()

print(f"{'Size':>6} | {'Time':>8} | {'OK':>6} | {'Fail':>6} | {'Fail%':>6} | {'img/s':>7}")
print("-" * 55)

for size in SIZES:
    dataset = full_dataset.select(range(size))
    success, fail = 0, 0
    t0 = time.time()

    for sample in dataset:
        url = sample["image_url"]
        try:
            with fsspec.open(url, "rb", expand=True) as f:
                img = PILImage.open(BytesIO(f.read())).convert("RGB")
            success += 1
        except Exception:
            fail += 1

    elapsed = time.time() - t0
    fail_pct = (fail / size) * 100
    throughput = success / elapsed if elapsed > 0 else 0

    print(f"{size:>6} | {elapsed:>7.1f}s | {success:>6} | {fail:>6} | {fail_pct:>5.1f}% | {throughput:>6.1f}/s")

print()
print("Done.")
