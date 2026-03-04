"""
Reproduce: VLM URL image loading with HF datasets.
Tests cast_column(Image()) vs manual download approaches.
Dataset: google-research-datasets/conceptual_captions (subset: labeled)
"""
from datasets import load_dataset, Image as datasets_Image, Dataset
from PIL import Image as PILImage
from io import BytesIO
from itertools import islice
import time

DATASET = "google-research-datasets/conceptual_captions"
SUBSET = "labeled"
SPLIT = "train"
N_SAMPLES = 20  # small slice for testing

print("=" * 60)
print("Loading dataset (streaming, first N samples)...")
print("=" * 60)
ds = load_dataset(DATASET, name=SUBSET, split=SPLIT, streaming=True)
rows = list(islice(ds, N_SAMPLES))
dataset = Dataset.from_list(rows)

print(f"Loaded {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
print(f"First image_url: {dataset[0]['image_url'][:100]}...")
print()

# ─── Test 1: cast_column(Image()) — what we tried ───
print("=" * 60)
print("TEST 1: cast_column(Image()) approach")
print("=" * 60)
try:
    ds_cast = dataset.cast_column("image_url", datasets_Image())
    success, fail = 0, 0
    t0 = time.time()
    for i, sample in enumerate(ds_cast):
        try:
            img = sample["image_url"]
            if img is not None:
                print(f"  [{i}] OK — {img.size} {img.mode}")
                success += 1
            else:
                print(f"  [{i}] None returned")
                fail += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            fail += 1
    elapsed = time.time() - t0
    print(f"\nResult: {success} ok, {fail} failed, {elapsed:.1f}s")
except Exception as e:
    print(f"CRASHED during iteration: {type(e).__name__}: {str(e)[:120]}")
print()

# ─── Test 2: Manual download with requests.Session ───
print("=" * 60)
print("TEST 2: requests.Session() approach")
print("=" * 60)
try:
    import requests
    session = requests.Session()
    success, fail = 0, 0
    t0 = time.time()
    for i, sample in enumerate(dataset):
        url = sample["image_url"]
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            img = PILImage.open(BytesIO(resp.content)).convert("RGB")
            print(f"  [{i}] OK — {img.size} {img.mode}")
            success += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            fail += 1
    elapsed = time.time() - t0
    print(f"\nResult: {success} ok, {fail} failed, {elapsed:.1f}s")
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {str(e)[:120]}")
print()

# ─── Test 3: urllib (stdlib) ───
print("=" * 60)
print("TEST 3: urllib approach (stdlib)")
print("=" * 60)
try:
    from urllib.request import urlopen, Request
    success, fail = 0, 0
    t0 = time.time()
    for i, sample in enumerate(dataset):
        url = sample["image_url"]
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                img = PILImage.open(BytesIO(resp.read())).convert("RGB")
            print(f"  [{i}] OK — {img.size} {img.mode}")
            success += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            fail += 1
    elapsed = time.time() - t0
    print(f"\nResult: {success} ok, {fail} failed, {elapsed:.1f}s")
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {str(e)[:120]}")
print()

# ─── Test 4: fsspec directly with expand=True ───
print("=" * 60)
print("TEST 4: fsspec.open() with expand=True")
print("=" * 60)
try:
    import fsspec
    success, fail = 0, 0
    t0 = time.time()
    for i, sample in enumerate(dataset):
        url = sample["image_url"]
        try:
            with fsspec.open(url, "rb", expand=True) as f:
                img = PILImage.open(BytesIO(f.read())).convert("RGB")
            print(f"  [{i}] OK — {img.size} {img.mode}")
            success += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            fail += 1
    elapsed = time.time() - t0
    print(f"\nResult: {success} ok, {fail} failed, {elapsed:.1f}s")
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {str(e)[:120]}")

print()
print("=" * 60)
print("DONE — compare success rates and timing above")
print("=" * 60)
